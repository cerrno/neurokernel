#!/usr/bin/env python

"""
Core Neurokernel classes.
"""

import atexit
import time

import bidict
from mpi4py import MPI
import numpy as np
import twiggy

from ctx_managers import IgnoreKeyboardInterrupt, OnKeyboardInterrupt, \
     ExceptionOnSignal, TryExceptionOnSignal
from mixins import LoggerMixin
import mpi
from tools.logging import setup_logger
from tools.misc import catch_exception, dtype_to_mpi
from tools.mpi import MPIOutput
from pattern import Interface, Pattern
from plsel import SelectorMethods
from pm import BasePortMapper, PortMapper
from routing_table import RoutingTable
from uid import uid

CTRL_TAG = 1

# MPI tags for distinguishing messages associated with different port types:
GPOT_TAG = CTRL_TAG+1
SPIKE_TAG = CTRL_TAG+2

class Module(mpi.Worker):
    """
    Processing module.

    This class repeatedly executes a work method until it receives a
    quit message via its control network port.

    Parameters
    ----------
    sel : str, unicode, or sequence
        Path-like selector describing the module's interface of
        exposed ports.
    sel_in, sel_out, sel_gpot, sel_spike : str, unicode, or sequence
        Selectors respectively describing all input, output, graded potential,
        and spiking ports in the module's interface.
    data_gpot, data_spike : numpy.ndarray
        Data arrays associated with the graded potential and spiking ports in
        the . Array length must equal the number
        of ports in a module's interface.
    columns : list of str
        Interface port attributes.
        Network port for controlling the module instance.
    ctrl_tag, gpot_tag, spike_tag : int
        MPI tags that respectively identify messages containing control data,
        graded potential port values, and spiking port values transmitted to 
        worker nodes.
    id : str
        Module identifier. If no identifier is specified, a unique
        identifier is automatically generated.
    device : int
        GPU device to use. May be set to None if the module does not perform
        GPU processing.
    routing_table : neurokernel.routing_table.RoutingTable
        Routing table describing data connections between modules. If no routing
        table is specified, the module will be executed in isolation.
    rank_to_id : bidict.bidict
        Mapping between MPI ranks and module object IDs.
    debug : bool
        Debug flag. When True, exceptions raised during the work method
        are not be suppressed.
    time_sync : bool
        Time synchronization flag. When True, debug messages are not emitted
        during module synchronization and the time taken to receive all incoming
        data is computed.

    Attributes
    ----------
    interface : Interface
        Object containing information about a module's ports.
    pm : dict
        `pm['gpot']` and `pm['spike']` are instances of neurokernel.pm.PortMapper that
        map a module's ports to the contents of the values in `data`.
    data : dict
        `data['gpot']` and `data['spike']` are arrays of data associated with 
        a module's graded potential and spiking ports.
    """

    def __init__(self, sel, sel_in, sel_out,
                 sel_gpot, sel_spike, data_gpot, data_spike,
                 columns=['interface', 'io', 'type'],
                 ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG, spike_tag=SPIKE_TAG,
                 id=None, device=None,
                 routing_table=None, rank_to_id=None,
                 debug=False, time_sync=False):

        super(Module, self).__init__(ctrl_tag)
        self.debug = debug
        self.time_sync = time_sync
        self.device = device

        self._gpot_tag = gpot_tag
        self._spike_tag = spike_tag

        # Require several necessary attribute columns:
        if 'interface' not in columns:
            raise ValueError('interface column required')
        if 'io' not in columns:
            raise ValueError('io column required')
        if 'type' not in columns:
            raise ValueError('type column required')

        # Manually register the file close method associated with MPIOutput
        # so that it is called by atexit before MPI.Finalize() (if the file is
        # closed after MPI.Finalize() is called, an error will occur):
        for k, v in twiggy.emitters.iteritems():
             if isinstance(v._output, MPIOutput):       
                 atexit.register(v._output.close)

        # Ensure that the input and output port selectors respectively
        # select mutually exclusive subsets of the set of all ports exposed by
        # the module:
        if not SelectorMethods.is_in(sel_in, sel):
            raise ValueError('input port selector not in selector of all ports')
        if not SelectorMethods.is_in(sel_out, sel):
            raise ValueError('output port selector not in selector of all ports')
        if not SelectorMethods.are_disjoint(sel_in, sel_out):
            raise ValueError('input and output port selectors not disjoint')

        # Ensure that the graded potential and spiking port selectors
        # respectively select mutually exclusive subsets of the set of all ports
        # exposed by the module:
        if not SelectorMethods.is_in(sel_gpot, sel):
            raise ValueError('gpot port selector not in selector of all ports')
        if not SelectorMethods.is_in(sel_spike, sel):
            raise ValueError('spike port selector not in selector of all ports')
        if not SelectorMethods.are_disjoint(sel_gpot, sel_spike):
            raise ValueError('gpot and spike port selectors not disjoint')

        # Save routing table and mapping between MPI ranks and module IDs:
        self.routing_table = routing_table
        self.rank_to_id = rank_to_id

        # Generate a unique ID if none is specified:
        if id is None:
            self.id = uid()
        else:

            # If a unique ID was specified and the routing table is not empty 
            # (i.e., there are connections between multiple modules),
            # the id must be a node in the table:
            if routing_table is not None and len(routing_table.ids) and \
                    not routing_table.has_node(id):
                raise ValueError('routing table must contain specified module ID')
            self.id = id

        # Reformat logger name:
        LoggerMixin.__init__(self, 'mod %s' % self.id)

        # Create module interface given the specified ports:
        self.interface = Interface(sel, columns)

        # Set the interface ID to 0; we assume that a module only has one interface:
        self.interface[sel, 'interface'] = 0

        # Set the port attributes:
        self.interface[sel_in, 'io'] = 'in'
        self.interface[sel_out, 'io'] = 'out'
        self.interface[sel_gpot, 'type'] = 'gpot'
        self.interface[sel_spike, 'type'] = 'spike'

        # Find the input and output ports:
        self.in_ports = self.interface.in_ports().to_tuples()
        self.out_ports = self.interface.out_ports().to_tuples()

        # Find the graded potential and spiking ports:
        self.gpot_ports = self.interface.gpot_ports().to_tuples()
        self.spike_ports = self.interface.spike_ports().to_tuples()

        self.in_gpot_ports = self.interface.in_ports().gpot_ports().to_tuples()
        self.in_spike_ports = self.interface.in_ports().spike_ports().to_tuples()
        self.out_gpot_ports = self.interface.out_ports().gpot_ports().to_tuples()
        self.out_spike_ports = self.interface.out_ports().spike_ports().to_tuples()

        # Set up mapper between port identifiers and their associated data:
        if len(data_gpot) != len(self.gpot_ports):
            raise ValueError('incompatible gpot port data array length')
        if len(data_spike) != len(self.spike_ports):
            raise ValueError('incompatible spike port data array length')
        self.data = {}
        self.data['gpot'] = data_gpot
        self.data['spike'] = data_spike
        self.pm = {}
        print self.data['spike']
        self.pm['gpot'] = PortMapper(sel_gpot, self.data['gpot'])
        self.pm['spike'] = PortMapper(sel_spike, self.data['spike'])

    def _init_gpu(self):
        """
        Initialize GPU device.

        Notes
        -----
        Must be called from within the `run()` method, not from within
        `__init__()`.
        """

        if self.device == None:
            self.log_info('no GPU specified - not initializing ')
        else:

            # Import pycuda.driver here so as to facilitate the
            # subclassing of Module to create pure Python LPUs that don't use GPUs:
            import pycuda.driver as drv
            drv.init()
            try:
                self.gpu_ctx = drv.Device(self.device).make_context()
            except Exception as e:
                self.log_info('_init_gpu exception: ' + e.message)
            else:
                atexit.register(self.gpu_ctx.pop)
                self.log_info('GPU initialized')

    @property
    def N_gpot_ports(self):
        """
        Number of exposed graded-potential ports.
        """

        return len(self.interface.gpot_ports())

    @property
    def N_spike_ports(self):
        """
        Number of exposed spiking ports.
        """

        return len(self.interface.spike_ports())

    def _get_in_data(self):
        """
        Get input data from incoming transmission buffer.

        Populate the data arrays associated with a module's ports using input
        data received from other modules.
        """

        if self.net in ['none', 'ctrl']:
            self.log_info('not retrieving from input buffer')
        else:
            self.log_info('retrieving from input buffer')

            # Since fan-in is not permitted, the data from all source modules
            # must necessarily map to different ports; we can therefore write each
            # of the received data to the array associated with the module's ports
            # here without worry of overwriting the data from each source module:
            for in_id in self._in_ids:
                # Check for exceptions so as to not fail on the first emulation
                # step when there is no input data to retrieve:
                try:

                    # The first entry of `data` contains graded potential values,
                    # while the second contains spiking port values (i.e., 0 or
                    # 1):
                    data = self._in_data[in_id].popleft()
                except:
                    self.log_info('no input data from [%s] retrieved' % in_id)
                else:
                    self.log_info('input data from [%s] retrieved' % in_id)

                    # Assign transmitted values directly to port data array:
                    if len(self._in_port_dict_ids['gpot'][in_id]):
                        self.pm['gpot'].set_by_inds(self._in_port_dict_ids['gpot'][in_id], data[0])
                    if len(self._in_port_dict_ids['spike'][in_id]):
                        self.pm['spike'].set_by_inds(self._in_port_dict_ids['spike'][in_id], data[1])
                    

    def _put_out_data(self):
        """
        Put specified output data in outgoing transmission buffer.

        Stage data from the data arrays associated with a module's ports for
        output to other modules.

        Notes
        -----
        The output spike port selection algorithm could probably be made faster.
        """

        if self.net in ['none', 'ctrl']:
            self.log_info('not populating output buffer')
        else:
            self.log_info('populating output buffer')

            # Clear output buffer before populating it:
            self._out_data = []

            # Select data that should be sent to each destination module and append
            # it to the outgoing queue:
            for out_id in self._out_ids:
                # Select port data using list of graded potential ports that can
                # transmit output:
                if len(self._out_port_dict_ids['gpot'][out_id]):
                    gpot_data = \
                        self.pm['gpot'].get_by_inds(self._out_port_dict_ids['gpot'][out_id])
                else:
                    gpot_data = np.array([], self.pm['gpot'].dtype)

                if len(self._out_port_dict_ids['spike'][out_id]):
                    spike_data = \
                        self.pm['spike'].get_by_inds(self._out_port_dict_ids['spike'][out_id])
            '''
            ACTUAL
                else:
                    spike_data = np.array([], self.pm['spike'].dtype)

                # Attempt to stage the emitted port data for transmission:            
                try:
                    self._out_data.append((out_id, (gpot_data, spike_data)))
            Modified: For some dumb reason, the system expects some spike data through ports
            '''
                try:
                    self._out_data.append((out_id, gpot_data))
                except:
                    self.log_info('no output data to [%s] sent' % out_id)
                else:
                    self.log_info('output data to [%s] sent' % out_id)
                
    def run_step(self):
        """
        Module work method.
    
        This method should be implemented to do something interesting with new 
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other 
        class attributes.
        """

        self.log_info('running execution step')

    def _init_port_dicts(self):
        """
        Initial dictionaries of source/destination ports in current module.
        """

        # Extract identifiers of source ports in the current module's interface
        # for all modules receiving output from the current module:
        self._out_port_dict = {}
        self._out_port_dict['gpot'] = {}
        self._out_port_dict['spike'] = {}
        self._out_port_dict_ids = {}
        self._out_port_dict_ids['gpot'] = {}
        self._out_port_dict_ids['spike'] = {}

        self._out_ids = self.routing_table.dest_ids(self.id)
        self._out_ranks = [self.rank_to_id[:i] for i in self._out_ids]
        for out_id in self._out_ids:
            self.log_info('extracting output ports for %s' % out_id)

            # Get interfaces of pattern connecting the current module to
            # destination module `out_id`; `int_0` is connected to the
            # current module, `int_1` is connected to the other module:
            pat = self.routing_table[self.id, out_id]['pattern']
            int_0 = self.routing_table[self.id, out_id]['int_0']
            int_1 = self.routing_table[self.id, out_id]['int_1']

            # Get ports in interface (`int_0`) connected to the current
            # module that are connected to the other module via the pattern:
            self._out_port_dict['gpot'][out_id] = \
                    pat.src_idx(int_0, int_1, 'gpot', 'gpot')
            self._out_port_dict_ids['gpot'][out_id] = \
                    self.pm['gpot'].ports_to_inds(self._out_port_dict['gpot'][out_id])
            self._out_port_dict['spike'][out_id] = \
                    pat.src_idx(int_0, int_1, 'spike', 'spike')
            self._out_port_dict_ids['spike'][out_id] = \
                    self.pm['spike'].ports_to_inds(self._out_port_dict['spike'][out_id])
           
        # Extract identifiers of destination ports in the current module's
        # interface for all modules sending input to the current module:
        self._in_port_dict = {}
        self._in_port_dict['gpot'] = {}
        self._in_port_dict['spike'] = {}
        self._in_port_dict_ids = {}
        self._in_port_dict_ids['gpot'] = {}
        self._in_port_dict_ids['spike'] = {}

        self._in_ids = self.routing_table.src_ids(self.id)
        self._in_ranks = [self.rank_to_id[:i] for i in self._in_ids]
        for in_id in self._in_ids:
            self.log_info('extracting input ports for %s' % in_id)

            # Get interfaces of pattern connecting the current module to
            # source module `in_id`; `int_1` is connected to the current
            # module, `int_0` is connected to the other module:
            pat = self.routing_table[in_id, self.id]['pattern']
            int_0 = self.routing_table[in_id, self.id]['int_0']
            int_1 = self.routing_table[in_id, self.id]['int_1']

            # Get ports in interface (`int_1`) connected to the current
            # module that are connected to the other module via the pattern:
            self._in_port_dict['gpot'][in_id] = \
                    pat.dest_idx(int_0, int_1, 'gpot', 'gpot')
            self._in_port_dict_ids['gpot'][in_id] = \
                    self.pm['gpot'].ports_to_inds(self._in_port_dict['gpot'][in_id])
            self._in_port_dict['spike'][in_id] = \
                    pat.dest_idx(int_0, int_1, 'spike', 'spike')
            self._in_port_dict_ids['spike'][in_id] = \
                    self.pm['spike'].ports_to_inds(self._in_port_dict['spike'][in_id])

    def _init_data_in(self):
        """
        Buffers for receiving data from other modules.

        Notes
        -----
        Must be executed after `_init_port_dicts()`.
        """

        # Allocate arrays for receiving data transmitted to the module so that
        # they don't have to be reallocated during every execution step
        # synchronization:
        self.data_in = {}
        self.data_in['gpot'] = {}
        self.data_in['spike'] = {}
        for in_id in self._in_ids:
            self.data_in['gpot'][in_id] = \
                np.empty(np.shape(self._in_port_dict['gpot'][in_id]), 
                         self.pm['gpot'].dtype)
            self.data_in['spike'][in_id] = \
                np.empty(np.shape(self._in_port_dict['spike'][in_id]), 
                         self.pm['spike'].dtype)
            
    def _sync(self):
        """
        Send output data and receive input data.
        """

        if self.time_sync:
            start = time.time()
        req = MPI.Request()
        requests = []

        # For each destination module, extract elements from the current
        # module's port data array, copy them to a contiguous array, and
        # transmit the latter:
        for dest_id, dest_rank in zip(self._out_ids, self._out_ranks):

            # Get source ports in current module that are connected to the
            # destination module:
            data_gpot = self.pm['gpot'].get_by_inds(self._out_port_dict_ids['gpot'][dest_id])
            data_spike = self.pm['spike'].get_by_inds(self._out_port_dict_ids['spike'][dest_id])

            if not self.time_sync:
                self.log_info('gpot data being sent to %s: %s' % \
                              (dest_id, str(data_gpot)))
                self.log_info('spike data being sent to %s: %s' % \
                              (dest_id, str(data_spike)))
            r = MPI.COMM_WORLD.Isend([data_gpot,
                                      dtype_to_mpi(data_gpot.dtype)],
                                     dest_rank, GPOT_TAG)
            requests.append(r)
            r = MPI.COMM_WORLD.Isend([data_spike,
                                      dtype_to_mpi(data_spike.dtype)],
                                     dest_rank, SPIKE_TAG)
            requests.append(r)

            if not self.time_sync:
                self.log_info('sending to %s' % dest_id)
        if not self.time_sync:
            self.log_info('sent all data from %s' % self.id)

        # For each source module, receive elements and copy them into the
        # current module's port data array:
        received_gpot = []
        received_spike = []
        ind_in_gpot_list = []
        ind_in_spike_list = []
        for src_id, src_rank in zip(self._in_ids, self._in_ranks):
            r = MPI.COMM_WORLD.Irecv([self.data_in['gpot'][src_id],
                                      dtype_to_mpi(data_gpot.dtype)],
                                     source=src_rank, tag=GPOT_TAG)
            requests.append(r)
            r = MPI.COMM_WORLD.Irecv([self.data_in['spike'][src_id],
                                      dtype_to_mpi(data_spike.dtype)],
                                     source=src_rank, tag=SPIKE_TAG)
            requests.append(r)
            if not self.time_sync:
                self.log_info('receiving from %s' % src_id)
        req.Waitall(requests)
        if not self.time_sync:
            self.log_info('received all data received by %s' % self.id)            

        # Copy received elements into the current module's data array:
        for src_id in self._in_ids:
            ind_in_gpot = self._in_port_dict_ids['gpot'][src_id]
            self.pm['gpot'].set_by_inds(ind_in_gpot, self.data_in['gpot'][src_id])
            ind_in_spike = self._in_port_dict_ids['spike'][src_id]
            self.pm['spike'].set_by_inds(ind_in_spike, self.data_in['spike'][src_id]) 

        # Save timing data:
        if self.time_sync:
            stop = time.time()
            n_gpot = 0
            n_spike = 0
            for src_id in self._in_ids:
                n_gpot += len(self.data_in['gpot'][src_id])
                n_spike += len(self.data_in['spike'][src_id])
            self.log_info('sent timing data to master')
            self.intercomm.isend(['sync_time',
                                  (self.rank, self.steps, start, stop,
                                   n_gpot*self.pm['gpot'].dtype.itemsize+\
                                   n_spike*self.pm['spike'].dtype.itemsize)],
                                 dest=0, tag=self._ctrl_tag)
        else:
            self.log_info('saved all data received by %s' % self.id)

    def run_step(self):
        """
        Module work method.

        This method should be implemented to do something interesting with new
        input port data in the module's `pm` attribute and update the attribute's
        output port data if necessary. It should not interact with any other
        class attributes.
        """

        self.log_info('running execution step')

    def pre_run(self):
        """
        Code to run before main loop.

        This method is invoked by the `run()` method before the main loop is
        started.
        """

        self.log_info('running code before body of worker %s' % self.rank)

        # Initialize _out_port_dict and _in_port_dict attributes:
        self._init_port_dicts()

        # Initialize data_in attribute:
        self._init_data_in()

        # Start timing the main loop:
        if self.time_sync:
            self.intercomm.isend(['start_time', (self.rank, time.time())],
                                 dest=0, tag=self._ctrl_tag)                
            self.log_info('sent start time to manager')

    def post_run(self):
        """
        Code to run after main loop.

        This method is invoked by the `run()` method after the main loop is
        started.
        """

        self.log_info('running code after body of worker %s' % self.rank)

        # Stop timing the main loop before shutting down the emulation:
        if self.time_sync:
            self.intercomm.isend(['stop_time', (self.rank, time.time())],
                                 dest=0, tag=self._ctrl_tag)

            self.log_info('sent stop time to manager')

        # Send acknowledgment message:
        self.intercomm.isend(['done', self.rank], 0, self._ctrl_tag)
        self.log_info('done message sent to manager')

    def run(self):
        """
        Body of process.
        """

        # Don't allow keyboard interruption of process:
        with IgnoreKeyboardInterrupt():

            # Activate execution loop:
            super(Module, self).run()

    def do_work(self):
        """
        Work method.

        This method is repeatedly executed by the Worker instance after the
        instance receives a 'start' control message and until it receives a 'stop'
        control message.
        """

        # If the debug flag is set, don't catch exceptions so that
        # errors will lead to visible failures:
        if self.debug:

            # Run the processing step:
            self.run_step()

            # Synchronize:
            self._sync()
        else:

            # Run the processing step:
            catch_exception(self.run_step, self.log_info)

            # Synchronize:
            catch_exception(self._sync, self.log_info)

class Manager(mpi.WorkerManager):
    """
    Module manager.

    Instantiates, connects, starts, and stops modules comprised by an
    emulation. All modules and connections must be added to a module manager
    instance before they can be run.

    Attributes
    ----------
    ctrl_tag : int
        MPI tag to identify control messages.
    modules : dict
        Module instances. Keyed by module object ID.
    routing_table : routing_table.RoutingTable
        Table of data transmission connections between modules.
    rank_to_id : bidict.bidict
        Mapping between MPI ranks and module object IDs.
    """

    def __init__(self, required_args=['sel', 'sel_in', 'sel_out',
                                      'sel_gpot', 'sel_spike'],
                 ctrl_tag=CTRL_TAG):
        super(Manager, self).__init__(ctrl_tag)

        # Required constructor args:
        self.required_args = required_args

        # One-to-one mapping between MPI rank and module ID:
        self.rank_to_id = bidict.bidict()

        # Unique object ID:
        self.id = uid()

        # Set up a dynamic table to contain the routing table:
        self.routing_table = RoutingTable()

        # Number of emulation steps to run:
        self.steps = np.inf

        # Variables for timing run loop:
        self.start_time = 0.0
        self.stop_time = 0.0

        # Variables for computing throughput:
        self.counter = 0
        self.total_sync_time = 0.0
        self.total_sync_nbytes = 0.0
        self.received_data = {}

        # Average step synchronization time:
        self._average_step_sync_time = 0.0

        # Computed throughput (only updated after an emulation run):    
        self._average_throughput = 0.0
        self._total_throughput = 0.0
        self.log_info('manager instantiated')

    @property
    def average_step_sync_time(self):
        """
        Average step synchronization time.
        """

        return self._average_step_sync_time
    @average_step_sync_time.setter
    def average_step_sync_time(self, t):
        self._average_step_sync_time = t

    @property
    def total_throughput(self):
        """
        Total received data throughput.
        """

        return self._total_throughput
    @total_throughput.setter
    def total_throughput(self, t):
        self._total_throughput = t

    @property
    def average_throughput(self):
        """
        Average received data throughput per step.
        """

        return self._average_throughput
    @average_throughput.setter
    def average_throughput(self, t):
        self._average_throughput = t

    def validate_args(self, target):
        """
        Check whether a class' constructor has specific arguments.

        Parameters
        ----------
        target : Module
            Module class to instantiate and run.
        
        Returns
        -------
        result : bool
            True if all of the required arguments are present, False otherwise.
        """

        arg_names = set(mpi.getargnames(target.__init__))
        for required_arg in self.required_args:
            if required_arg not in arg_names:
                return False
        return True

    def add(self, target, id, *args, **kwargs):
        """
        Add a module class to the emulation.

        Parameters
        ----------
        target : Module
            Module class to instantiate and run.
        id : str
            Identifier to use when connecting an instance of this class
            with an instance of some other class added to the emulation.
        args : sequence
            Sequential arguments to pass to the constructor of the class
            associated with identifier `id`.
        kwargs : dict
            Named arguments to pass to the constructor of the class
            associated with identifier `id`.
        """

        if not issubclass(target, Module):
            raise ValueError('target class is not a Module subclass')
        argnames = mpi.getargnames(target.__init__)

        # Selectors must be passed to the module upon instantiation;
        # the module manager must know about them to assess compatibility:
        # XXX: keep this commented out for the time being because it interferes
        # with instantiation of child classes (such as those in LPU.py):
        # if not self.validate_args(target):
        #    raise ValueError('class constructor missing required args')

        # Need to associate an ID and the routing table with each module class
        # to instantiate; because the routing table's can potentially occupy
        # lots of space, we don't add it to the argument dict here - it is
        # broadcast to all processes separately and then added to the argument
        # dict in mpi_backend.py:
        kwargs['id'] = id
        kwargs['rank_to_id'] = self.rank_to_id
        rank = super(Manager, self).add(target, *args, **kwargs)
        self.rank_to_id[rank] = id
 
    def connect(self, id_0, id_1, pat, int_0=0, int_1=1, compat_check=True):
        if not isinstance(pat, Pattern):
            raise ValueError('pat is not a Pattern instance')
        if id_0 not in self.rank_to_id.values():
            raise ValueError('unrecognized module id %s' % id_0)
        if id_1 not in self.rank_to_id.values():
            raise ValueError('unrecognized module id %s' % id_1)
        if not (int_0 in pat.interface_ids and int_1 in pat.interface_ids):
            raise ValueError('unrecognized pattern interface identifiers')
        self.log_info('connecting modules {0} and {1}'
                      .format(id_0, id_1))

        # Check compatibility of the interfaces exposed by the modules and the
        # pattern; since the manager only contains module classes and not class
        # instances, we need to create Interface instances from the selectors
        # associated with the modules in order to test their compatibility:
        if compat_check:
            rank_0 = self.rank_to_id.inv[id_0]
            rank_1 = self.rank_to_id.inv[id_1]

            self.log_info('checking compatibility of modules {0} and {1} and'
                             ' assigned pattern'.format(id_0, id_1))
            mod_int_0 = Interface(self._kwargs[rank_0]['sel'])
            mod_int_0[self._kwargs[rank_0]['sel']] = 0
            mod_int_1 = Interface(self._kwargs[rank_1]['sel'])
            mod_int_1[self._kwargs[rank_1]['sel']] = 0

            mod_int_0[self._kwargs[rank_0]['sel_in'], 'io'] = 'in'
            mod_int_0[self._kwargs[rank_0]['sel_out'], 'io'] = 'out'
            mod_int_0[self._kwargs[rank_0]['sel_gpot'], 'type'] = 'gpot'
            mod_int_0[self._kwargs[rank_0]['sel_spike'], 'type'] = 'spike'
            mod_int_1[self._kwargs[rank_1]['sel_in'], 'io'] = 'in'
            mod_int_1[self._kwargs[rank_1]['sel_out'], 'io'] = 'out'
            mod_int_1[self._kwargs[rank_1]['sel_gpot'], 'type'] = 'gpot'
            mod_int_1[self._kwargs[rank_1]['sel_spike'], 'type'] = 'spike'

            if not mod_int_0.is_compatible(0, pat.interface, int_0, True):
                raise ValueError('module %s interface incompatible '
                                 'with pattern interface %s' % (id_0, int_0))
            if not mod_int_1.is_compatible(0, pat.interface, int_1, True):
                raise ValueError('module %s interface incompatible '
                                 'with pattern interface %s' % (id_1, int_1))

        # XXX Need to check for fan-in XXX

        # Store the pattern information in the routing table:
        self.log_info('updating routing table with pattern')
        if pat.is_connected(0, 1):
            self.routing_table[id_0, id_1] = {'pattern': pat,
                                              'int_0': int_0, 'int_1': int_1}
        if pat.is_connected(1, 0):
            self.routing_table[id_1, id_0] = {'pattern': pat,
                                              'int_0': int_1, 'int_1': int_0}

        self.log_info('connected modules {0} and {1}'.format(id_0, id_1))

    def process_worker_msg(self, msg):

        # Process timing data sent by workers:
        if msg[0] == 'start_time':
            rank, self.start_time = msg[1]
            self.log_info('start time data: %s' % str(msg[1]))
        elif msg[0] == 'stop_time':
            rank, self.stop_time = msg[1]
            self.log_info('stop time data: %s' % str(msg[1]))
        elif msg[0] == 'sync_time':
            rank, steps, start, stop, nbytes = msg[1]
            self.log_info('sync time data: %s' % str(msg[1]))

            # Collect timing data for each execution step:
            if steps not in self.received_data:
                self.received_data[steps] = {}                    
            self.received_data[steps][rank] = (start, stop, nbytes)

            # After adding the latest timing data for a specific step, check
            # whether data from all modules has arrived for that step:
            if set(self.received_data[steps].keys()) == set(self.rank_to_id.keys()):

                # Exclude the very first step to avoid including delays due to
                # PyCUDA kernel compilation:
                if steps != 0:

                    # The duration an execution is assumed to be the longest of
                    # the received intervals:
                    step_sync_time = max([(d[1]-d[0]) for d in self.received_data[steps].values()])

                    # Obtain the total number of bytes received by all of the
                    # modules during the execution step:
                    step_nbytes = sum([d[2] for d in self.received_data[steps].values()])

                    self.total_sync_time += step_sync_time
                    self.total_sync_nbytes += step_nbytes

                    self.average_throughput = (self.average_throughput*self.counter+\
                                              step_nbytes/step_sync_time)/(self.counter+1)
                    self.average_step_sync_time = (self.average_step_sync_time*self.counter+\
                                                   step_sync_time)/(self.counter+1)

                    self.counter += 1
                else:
                    # To skip the first sync step, set the start time to the
                    # latest stop time of the first step:
                    self.start_time = max([d[1] for d in self.received_data[steps].values()])

                # Clear the data for the processed execution step so that
                # that the received_data dict doesn't consume unnecessary memory:
                del self.received_data[steps]

            # Compute throughput using accumulated timing data:
            if self.total_sync_time > 0:
                self.total_throughput = self.total_sync_nbytes/self.total_sync_time
            else:
                self.total_throughput = 0.0

    def wait(self):
        super(Manager, self).wait()
        self.log_info('avg step sync time/avg per-step throughput' \
                      '/total transm throughput/run loop duration:' \
                      '%s, %s, %s, %s' % \
                      (self.average_step_sync_time, self.average_throughput, 
                       self.total_throughput, self.stop_time-self.start_time))
        
if __name__ == '__main__':
    import neurokernel.mpi_relaunch

    class MyModule(Module):
        """
        Example of derived module class.
        """

        def run_step(self):

            super(MyModule, self).run_step()

            # Do something with input graded potential data:
            self.log_info('input gpot port data: '+str(self.pm['gpot'][self.in_gpot_ports]))

            # Do something with input spike data:
            self.log_info('input spike port data: '+str(self.pm['spike'][self.in_spike_ports]))

            # Output random graded potential data:
            out_gpot_data = np.random.rand(len(self.out_gpot_ports))
            self.pm['gpot'][self.out_gpot_ports] = out_gpot_data
            self.log_info('output gpot port data: '+str(out_gpot_data))

            # Randomly select output ports to emit spikes:
            out_spike_data = np.random.randint(0, 2, len(self.out_spike_ports))
            self.pm['spike'][self.out_spike_ports] = out_spike_data
            self.log_info('output spike port data: '+str(out_spike_data))

    logger = mpi.setup_logger(screen=True, file_name='neurokernel.log',
                              mpi_comm=MPI.COMM_WORLD, multiline=True)

    man = Manager()

    m1_int_sel_in_gpot = '/a/in/gpot0,/a/in/gpot1'
    m1_int_sel_out_gpot = '/a/out/gpot0,/a/out/gpot1'
    m1_int_sel_in_spike = '/a/in/spike0,/a/in/spike1'
    m1_int_sel_out_spike = '/a/out/spike0,/a/out/spike1'
    m1_int_sel = ','.join([m1_int_sel_in_gpot, m1_int_sel_out_gpot,
                           m1_int_sel_in_spike, m1_int_sel_out_spike])
    m1_int_sel_in = ','.join((m1_int_sel_in_gpot, m1_int_sel_in_spike))
    m1_int_sel_out = ','.join((m1_int_sel_out_gpot, m1_int_sel_out_spike))
    m1_int_sel_gpot = ','.join((m1_int_sel_in_gpot, m1_int_sel_out_gpot))
    m1_int_sel_spike = ','.join((m1_int_sel_in_spike, m1_int_sel_out_spike))
    N1_gpot = SelectorMethods.count_ports(m1_int_sel_gpot)
    N1_spike = SelectorMethods.count_ports(m1_int_sel_spike)

    m2_int_sel_in_gpot = '/b/in/gpot0,/b/in/gpot1'
    m2_int_sel_out_gpot = '/b/out/gpot0,/b/out/gpot1'
    m2_int_sel_in_spike = '/b/in/spike0,/b/in/spike1'
    m2_int_sel_out_spike = '/b/out/spike0,/b/out/spike1'
    m2_int_sel = ','.join([m2_int_sel_in_gpot, m2_int_sel_out_gpot,
                           m2_int_sel_in_spike, m2_int_sel_out_spike])
    m2_int_sel_in = ','.join((m2_int_sel_in_gpot, m2_int_sel_in_spike))
    m2_int_sel_out = ','.join((m2_int_sel_out_gpot, m2_int_sel_out_spike))
    m2_int_sel_gpot = ','.join((m2_int_sel_in_gpot, m2_int_sel_out_gpot))
    m2_int_sel_spike = ','.join((m2_int_sel_in_spike, m2_int_sel_out_spike))
    N2_gpot = SelectorMethods.count_ports(m2_int_sel_gpot)
    N2_spike = SelectorMethods.count_ports(m2_int_sel_spike)

    # Note that the module ID doesn't need to be listed in the specified
    # constructor arguments:
    m1_id = 'm1   '
    man.add(MyModule, m1_id, m1_int_sel, m1_int_sel_in, m1_int_sel_out,
            m1_int_sel_gpot, m1_int_sel_spike,
            np.zeros(N1_gpot, dtype=np.double),
            np.zeros(N1_spike, dtype=int),
            ['interface', 'io', 'type'],
            CTRL_TAG, GPOT_TAG, SPIKE_TAG, time_sync=True)
    m2_id = 'm2   '
    man.add(MyModule, m2_id, m2_int_sel, m2_int_sel_in, m2_int_sel_out,
            m2_int_sel_gpot, m2_int_sel_spike,
            np.zeros(N2_gpot, dtype=np.double),
            np.zeros(N2_spike, dtype=int),
            ['interface', 'io', 'type'],
            CTRL_TAG, GPOT_TAG, SPIKE_TAG, time_sync=True)

    # Make sure that all ports in the patterns' interfaces are set so 
    # that they match those of the modules:
    pat12 = Pattern(m1_int_sel, m2_int_sel)
    pat12.interface[m1_int_sel_out_gpot] = [0, 'in', 'gpot']
    pat12.interface[m1_int_sel_in_gpot] = [0, 'out', 'gpot']
    pat12.interface[m1_int_sel_out_spike] = [0, 'in', 'spike']
    pat12.interface[m1_int_sel_in_spike] = [0, 'out', 'spike']
    pat12.interface[m2_int_sel_in_gpot] = [1, 'out', 'gpot']
    pat12.interface[m2_int_sel_out_gpot] = [1, 'in', 'gpot']
    pat12.interface[m2_int_sel_in_spike] = [1, 'out', 'spike']
    pat12.interface[m2_int_sel_out_spike] = [1, 'in', 'spike']
    pat12['/a/out/gpot0', '/b/in/gpot0'] = 1
    pat12['/a/out/gpot1', '/b/in/gpot1'] = 1
    pat12['/b/out/gpot0', '/a/in/gpot0'] = 1
    pat12['/b/out/gpot1', '/a/in/gpot1'] = 1
    pat12['/a/out/spike0', '/b/in/spike0'] = 1
    pat12['/a/out/spike1', '/b/in/spike1'] = 1
    pat12['/b/out/spike0', '/a/in/spike0'] = 1
    pat12['/b/out/spike1', '/a/in/spike1'] = 1
    man.connect(m1_id, m2_id, pat12, 0, 1)

    # Start emulation and allow it to run for a little while before shutting
    # down.  To set the emulation to exit after executing a fixed number of
    # steps, start it as follows and remove the sleep statement:
    # man.start(500)
    man.spawn()
    man.start(20)
    man.wait()
