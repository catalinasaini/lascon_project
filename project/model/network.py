import nest

class Network:
    """
        The Network class serves as base class to create the thalamo-cortical network and input external signals.
    """
    
    def __init__(
        self, cx_population
    ):
        """
        Network creation
        Create AEIF (Adaptative Exponential Integrate-and-Fire) alpha neurons population for the populations:

        cx: pyramidal neurons (+) [in the cortex]
        in: inhibitory interneurons (-) [in the cortex]
        tc: thalamic relay neurons (+) [thalamocortical neurons]
        re: reticular neurons (-) [thalamic]
        
        :param cx_population: Size of the pyramidal neuron (cx) population.
        :type cx_population: int        
        """
        
        assert isinstance(cx_population, int), "No size of the pyramidal neuron (cx) population provided."
        
        # Declare params of static synapses
        W_IN_CX  = -4
        W_CX_IN = 60
        W_TC_RE = 10
        W_RE_TC = -10
        W_IN_IN = -1
        W_RE_RE = -1
        
        # Declare params of STDP synapses
        W_MAX_CXCX = 150                        # Max weight value fo the cx-cx connection: 150
        W_MAX_CXTC = 130                        # Max weight value fo the cx-tc connection: 130 
        W_MAX_TCCX = 5.5                        # Max weight value fo the tc-cx connection 
        W_INIT_INT = 1                          # Initial weight value for the cx-cx and cx-tc populations
        W_INIT_FLOAT = 1.0                      # Initial weight value for the tc-cx population
        ALPHA_SYM = 1.0                         # Alpha of the symmetric STDP synapse
        
        # Declare number of populations
        self.IN_N = 200
        self.TC_N = 324              # The number of thalamic neurons is the same as the dimension of the feature vector produced by the pre-processing of visual input
        self.RE_N = 200
        self.cx_n = cx_population    # Groups of 20 neurons for each image in the training set. In a first set of runs, the training set was composed of 9 images.

        # Declate set of cx neurons
        self.SET_CX_NEURON = 20          # Groups of 20 neurons for each image in the training set.
        
        # Declare Poisson generator signal
        self.sleep_osc = None
        
        # Declare devices
        self.mult_cx = None
        self.mult_in = None 
        self.mult_tc = None 
        self.mult_re = None 
        self.spikes_cx = None 
        self.spikes_in = None 
        self.spikes_tc = None 
        self.spikes_re = None         
        
        # Change V_peak and b params accordingly to the paper
        V_peak = nest.GetDefaults('aeif_cond_alpha')['V_th'] + 5 * nest.GetDefaults('aeif_cond_alpha')['Delta_T']
        neuron_params = {"V_peak": V_peak, "b": 0.01}

        nest.SetDefaults('aeif_cond_alpha', neuron_params)

        # Creating populations
        self.cx_pop = nest.Create('aeif_cond_alpha', self.cx_n)
        self.IN_POP = nest.Create('aeif_cond_alpha', self.IN_N)
        self.TC_POP = nest.Create('aeif_cond_alpha', self.TC_N)
        self.RE_POP = nest.Create('aeif_cond_alpha', self.RE_N)
        
        # Connect populations with static synapses
        nest.Connect(self.IN_POP, self.cx_pop, syn_spec={"weight": W_IN_CX}) # inhibitory interneurons -> pyramidal neurons
        nest.Connect(self.cx_pop, self.IN_POP, syn_spec={"weight": W_CX_IN}) # pyramidal neurons -> inhibitory interneurons
        nest.Connect(self.TC_POP, self.RE_POP, syn_spec={"weight": W_TC_RE}) # thalamic relay -> reticular neurons
        nest.Connect(self.RE_POP, self.TC_POP, syn_spec={"weight": W_RE_TC}) # reticular neurons -> thalamic relay
        nest.Connect(self.IN_POP, self.IN_POP, syn_spec={"weight": W_IN_IN}) # inhibitory interneurons -> inhibitory interneuros
        nest.Connect(self.RE_POP, self.RE_POP, syn_spec={"weight": W_RE_RE}) # reticular neurons -> reticular neurons
        
        # Connect populations with STDP synapses
        self.syn_dict_cxcx = {"synapse_model": "stdp_synapse", 
                            "alpha": ALPHA_SYM,
                            "weight": W_INIT_INT,
                            "Wmax": W_MAX_CXCX}

        self.syn_dict_cxtc = {"synapse_model": "stdp_synapse", 
                            "alpha": ALPHA_SYM,
                            "weight": W_INIT_INT,
                            "Wmax": W_MAX_CXTC}

        self.syn_dict_tccx = {"synapse_model": "stdp_synapse", 
                            "alpha": ALPHA_SYM,
                            "weight": W_INIT_FLOAT,
                            "Wmax": W_MAX_TCCX}

        # Connect populations
        nest.Connect(self.cx_pop, self.cx_pop, syn_spec=self.syn_dict_cxcx)         # Cx -> Cx
        nest.Connect(self.cx_pop, self.TC_POP, syn_spec=self.syn_dict_cxtc)         # Cx -> Tc
        nest.Connect(self.TC_POP, self.cx_pop, syn_spec=self.syn_dict_tccx)         # TC -> Cx
    
    def create_context_signal(self, time_id): 
        """
        Create the contextual signal using Poissan generator.
        
        :param time_id: it defines the start time of the Poisson signal. For example, time_id = 0 makes the signal starts at 0.
                        time_id = 1 makes it start after 900 ms, i.e., the duration of the signal plus a quiescent period.
        :type: int

        Returns:
            Obj: returns the Poissan generator object.
        """
        # Assert argument is valid
        assert (time_id <= self.cx_n//self.SET_CX_NEURON), f"Type a value between 0 and {int(self.cx_n//self.SET_CX_NEURON)}."
        assert (time_id >= 0), f"Type a value higher than 0."
        assert isinstance(time_id, int), "Type an int value."
        
        # Declare variables
        CONTEXT_RATE = 2000.0                       # Hz
        SIGN_DUR = 450                              # Duration of contextual signal in ms
        time_start = time_id * SIGN_DUR * 2
        time_stop = time_start + SIGN_DUR           # Set time stop of Poisson generator
        
        # Generate contextual signal
        context_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        context_sign.set(rate=CONTEXT_RATE, start=time_start, stop=time_stop)
        
        # Display result
        print("Contextual signal successfully created.")
                
        #  Return Pooisson generator
        return context_sign
    
    def create_inhib_signal(self, time_id): 
        """
        Create the inhibitory signal using Poissan generator.
        
        :param time_id: it defines the start time of the Poisson signal. For example, time_id = 0 makes the signal starts at 0.
                        time_id = 1 makes it start after 900 ms, i.e., the duration of the signal plus a quiescent period.
        :type: int

        Returns:
            Obj: returns the Poissan generator object.
        """
        # Assert argument is valid
        assert (time_id <= self.cx_n//self.SET_CX_NEURON), f"Type a value between 0 and {int(self.cx_n//self.SET_CX_NEURON)}."
        assert (time_id >= 0), f"Type a value higher than 0."
        assert isinstance(time_id, int), "Type an int value."
        
        # Declare variables
        INHIB_RATE = 10000.0                        # Hz
        SIGN_DUR = 450                              # Duration of inhibitory signal in ms
        time_start = time_id * SIGN_DUR * 2
        time_stop = time_start + SIGN_DUR           # Set time stop of Poisson generator
        
        # Generate inhibitory signal
        inhib_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        inhib_sign.set(rate=INHIB_RATE, start=time_start, stop=time_stop)
        
        # Display result
        print("Inhibitory signal successfully created.")
                
        #  Return Pooisson generator
        return inhib_sign

    def create_train_signal(self, time_id): 
        """
        Create the training signal using Poissan generator.
        
        :param time_id: it defines the start time of the Poisson signal. For example, time_id = 0 makes the signal starts at 0.
                        time_id = 1 makes it start after 900 ms, i.e., the duration of the signal plus a quiescent period.
        :type: int

        Returns:
            Obj: returns the Poissan generator object.
        """
        # Assert argument is valid
        assert (time_id <= self.cx_n//self.SET_CX_NEURON), f"Type a value between 0 and {int(self.cx_n//self.SET_CX_NEURON)}."
        assert (time_id >= 0), f"Type a value higher than 0."
        assert isinstance(time_id, int), "Type an int value."
        
        # Declare variables
        TRAIN_RATE = 30000.0                        # Hz
        SIGN_DUR = 650                              # Duration of training signal in ms
        time_start = time_id * SIGN_DUR * 2
        time_stop = time_start + SIGN_DUR           # Set time stop of Poisson generator
        
        # Generate training signal
        train_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        train_sign.set(rate=TRAIN_RATE, start=time_start, stop=time_stop)
        
        # Display result
        print("Training signal successfully created.")
                
        #  Return Pooisson generator
        return train_sign
    
    def input_context_signal(self, neuron_group):
        """
        Every time a new training image is presented to the network through the thalamic pathway, the facilitation signal 
        coming from the contextual signal provides a 2 kHz Poisson spike train to a different set of 20 neurons, inducing 
        the group to encode for that specific input stimulus.
        
        Turned off during the retrieval phase.
        
        :param neuron_group: Parameter that defines the slicing of the cx population. For example, neuron_group_id=1 slices 
                                from 0:20; neuron_group_id=2 slices from 20:40. Also, it defines the time_id for the function 
                                create_context_signal, since we are inputting different-time signals to the sliced neuronal populations.
        :param type: int
        """
        # Declare variables
        WEIGHT_SIGN_CX = 15          # Weight of connection between contextual signal and cx population
        
        # Define set of neurons
        size_group = self.SET_CX_NEURON * neuron_group
        
        assert (size_group <= self.cx_n), f"Type a value between 0 and {int(self.cx_n//self.SET_CX_NEURON)}."
        assert (neuron_group >= 0), f"Type a value between 0 and {int(self.cx_n//self.SET_CX_NEURON)}."
        assert isinstance(neuron_group, int), "Type an int value."
        
        # Define slicing
        start_slice = neuron_group * self.SET_CX_NEURON
        end_slice =  start_slice + self.SET_CX_NEURON
        
        # Get contextual signal
        context_sign = self.create_context_signal(neuron_group)
        
        # Connect them to the neurons
        print("Connecting input to the cx population...")
        nest.Connect(context_sign, self.cx_pop[start_slice:end_slice], syn_spec={"weight": WEIGHT_SIGN_CX})
        
        # Display connection
        print("... contextual signal successfully connected to the cx population.")
    
    def input_inhib_signal(self, time_id):
        """
        A 10 kHz Poisson spike train is provided to inhibitory neurons to prevent already trained neurons to respond 
        to new stimuli in the training phase.
        
        Only input it after the first training set. Turned off during the retrieval phase.
        
        :param time_id: Parameter that defines the slicing of the cx population. For example, neuron_group_id=1 slices 
                                from 0:20; neuron_group_id=2 slices from 20:40. Also, it defines the time_id for the function 
                                create_context_signal, since we are inputting different-time signals to the sliced neuronal populations.
        :param type: int
        """
        # Declare variables
        WEIGHT_INH_IN = 5              # Weight of inhibitory signal to in population
        
        # Generate inhibitory signal
        inhib_sign = self.create_inhib_signal(time_id)    
        
        # Connect them to the neurons
        print("Connecting input to the in population...")
        nest.Connect(inhib_sign, self.IN_POP, syn_spec={"weight": WEIGHT_INH_IN})   
        
        # Display connection
        print("... inhibitory signal successfully connected to the in population.")     

    def input_train_signal(self, time_id, feature_vector):
        """
        During the retrieval phase only the 30 kHz input to thalamic cell is provided, while the contextual signal is off.
        
        :param time_id: Parameter that defines the slicing of the cx population. For example, neuron_group_id=1 slices 
                                from 0:20; neuron_group_id=2 slices from 20:40. Also, it defines the time_id for the function 
                                create_context_signal, since we are inputting different-time signals to the sliced neuronal populations.
        :param type: int
        
        :param feature vector: A binary list of the TC_POP size length (i.e., 324)
        :type: list
        """
        # Variables
        WEIGHT_TRAIN_TC = 8                                         # Weight of Poisson to tc population
        time_id = time_id                                           # Iterator generating the training signal
        feature_vector=feature_vector                               # Feature vector from argument
        
        # Generate training signal
        train_sign = self.create_train_signal(time_id)

        # Connect training signal to neurons based on the feature vector
        print("Connecting input to the tc population...")
        for i, apply_input in enumerate(feature_vector):
            if apply_input:
                nest.Connect(train_sign, self.TC_POP[i], syn_spec={"weight": WEIGHT_TRAIN_TC})
        
        # Display connection
        print("... training signal successfully connected to the tc population.")
        
    def input_sleep(self):
        """
        After the training stage, the sleep-like thalamo-cortical spontaneous slow oscillations activity is induced for 
        a total duration of 600s by providing a non-specific Poisson noise inside the cortex (700 Hz) and increasing the 
        strength of SFA parameter (b=60, in eq. (1)). No external stimulus is provided to cells. Also, the synaptic weights between 
        inhibitory and excitatory neurons in the cortex is reduced to -0.5. In this stage asymmetric STDP plasticity (alpha=3.0) is active in 
        the recurrent cx connectivity, inducing sleep-induced modification in the synaptic weights structure. The parameters' change to 
        obtain the slow oscillating regime were chosen relying on mean field theory framework18,19.
        """
        # Variables
        OSC_RATE = 700.0                                    # Hz
        SLEEP_DUR = 600000.0                                # Sleep duration in ms
        NEW_WEIGHT_IN_CX = -0.5                             # Weight of the synapse between in -> cx population
        ALPHA_ASSYM = 3.0                                   # Alpha for the assymetric STDP plasticity stage
        b = 60                                              # SFA parameter
        params_sleep = {"individual_spike_trains": False}   # False: the generator sends the same spike train to all of its targets
        
        # Change the individual_spike_trains from True (default) to False
        print("Generating sleep oscillations...")
        nest.CopyModel("sinusoidal_poisson_generator", "sinusoidal_sleep", params=params_sleep)
        
        # Create sleep oscillation
        self.sleep_osc = nest.Create("sinusoidal_sleep")
        
        # Set frequencies
        self.sleep_osc.set(rate=OSC_RATE, stop=SLEEP_DUR)
        print("...done.")
               
        # Set the SFA parameters
        sfa_params = {"b": b}

        # Apply the initial parameters to the population
        nest.SetStatus(self.cx_pop, sfa_params)
        
        # Change connection weight W_in_cx to -0.5 and alpha to 3.0
        # Get Connections
        syn_in_cx = nest.GetConnections(self.IN_POP, self.cx_pop, synapse_model='static_synapse')
        syn_cx_cx = nest.GetConnections(self.cx_pop, self.cx_pop, synapse_model='stdp_synapse')

        # Apply the new parameters to the synapse
        syn_in_cx.set({"weight": NEW_WEIGHT_IN_CX})
        syn_cx_cx.set({"alpha": ALPHA_ASSYM})
        
        # Connect sleep oscillation to the neurons
        print("Connecting input to the cortex populations...")
        nest.Connect(self.sleep_osc, self.cx_pop)
        nest.Connect(self.sleep_osc, self.IN_POP)
        
        # Display connection
        print("... sleep oscillation signal successfully inputed to the cx and in populations (i.e., whole cortex).")      
   
    def set_multimeters(self):
        """
        Create multimeters to all populations.
        
        :return: four multimeters respectively to the cx, in, tc, and re populations.
        """
        # cx population
        self.mult_cx = nest.Create("multimeter")
        self.mult_cx.set(record_from=["V_m"])

        # in
        self.mult_in = nest.Create("multimeter")
        self.mult_in.set(record_from=["V_m"])

        # tc
        self.mult_tc = nest.Create("multimeter")
        self.mult_tc.set(record_from=["V_m"])

        # re
        self.mult_re = nest.Create("multimeter")
        self.mult_re.set(record_from=["V_m"])
        
    def set_spike_recorders(self):
        """
        Create spike records to all populations.
        
        :return: four spike recorders respectively to the cx, in, tc, and re populations.
        """
        # cx population
        self.spikes_cx = nest.Create("spike_recorder")

        # in
        self.spikes_in = nest.Create("spike_recorder")
        
        # tc
        self.spikes_tc = nest.Create("spike_recorder")

        # re
        self.spikes_re = nest.Create("spike_recorder")
    
    def connect_all_devices(self):
        """
        Connect multimeters and spike recorders created with set_multimeters() and set_spike_recorders().
        """
        # Get multimeters
        self.set_multimeters()
        self.set_spike_recorders()
        
        # Connect the multimeter to all populations
        nest.Connect(self.mult_cx, self.cx_pop)
        nest.Connect(self.mult_in, self.IN_POP)
        nest.Connect(self.mult_tc, self.TC_POP)
        nest.Connect(self.mult_re, self.RE_POP)

        # Connect the spike_recorder to all populations
        nest.Connect(self.cx_pop, self.spikes_cx)
        nest.Connect(self.IN_POP, self.spikes_in)
        nest.Connect(self.TC_POP, self.spikes_tc)
        nest.Connect(self.RE_POP, self.spikes_re)
        

