import nest

class Network:
    """
        The Network class serves as base class to create the thalamo-cortical network, modify its synapse weights and input external signals.
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

        # Define number of populations
        self.IN_N = 200
        self.TC_N = 324              # The number of thalamic neurons is the same as the dimension of the feature vector produced by the pre-processing of visual input
        self.RE_N = 200
        self.cx_n = cx_population    # Groups of 20 neurons for each image in the training set. In a first set of runs, the training set was composed of 9 images.

        # Change V_peak accordingly to the paper
        V_peak = nest.GetDefaults('aeif_cond_alpha')['V_th'] + 5 * nest.GetDefaults('aeif_cond_alpha')['Delta_T']
        neuron_params = {"V_peak": V_peak}

        nest.SetDefaults('aeif_cond_alpha', neuron_params)

        # Creating populations
        self.IN_POP = nest.Create('aeif_cond_alpha', self.IN_N)
        self.TC_POP = nest.Create('aeif_cond_alpha', self.TC_N)
        self.RE_POP = nest.Create('aeif_cond_alpha', self.RE_N)
        self.cx_pop = nest.Create('aeif_cond_alpha', self.cx_n)
        
        # Connect populations with static synapses
        nest.Connect(self.IN_POP, self.cx_pop, syn_spec={"weight": -4}) # inhibitory interneurons -> pyramidal neurons
        nest.Connect(self.cx_pop, self.IN_POP, syn_spec={"weight": 60}) # pyramidal neurons -> inhibitory interneurons
        nest.Connect(self.TC_POP, self.RE_POP, syn_spec={"weight": 10}) # thalamic relay -> reticular neurons
        nest.Connect(self.RE_POP, self.TC_POP, syn_spec={"weight": -10}) # reticular neurons -> thalamic relay
        nest.Connect(self.IN_POP, self.IN_POP, syn_spec={"weight": -1}) # inhibitory interneurons -> inhibitory interneuros
        nest.Connect(self.RE_POP, self.RE_POP, syn_spec={"weight": -1}) # reticular neurons -> reticular neurons
        
        # Connect populations with STDP synapses
        self.W_MAX_CXCX = 15                        # Max weight value fo the cx-cx connection: 150. I've changed to 15 to make it run for the tests.     
        self.W_MAX_CXTC = 3                        # Max weight value fo the cx-tc connection: 130 I've changed to 13 to make it run for the tests.
        self.W_MAX_TCCX = 5.5                        # Max weight value fo the tc-cx connection 
        self.syn_alpha = 1.0

        self.syn_dict_cxcx = {"synapse_model": "stdp_synapse", 
                            "alpha": self.syn_alpha,
                            "weight": 1,
                            "Wmax": self.W_MAX_CXCX}

        self.syn_dict_cxtc = {"synapse_model": "stdp_synapse", 
                            "alpha": self.syn_alpha,
                            "weight": 1,
                            "Wmax": self.W_MAX_CXTC}

        self.syn_dict_tccx = {"synapse_model": "stdp_synapse", 
                            "alpha": self.syn_alpha,
                            "weight": 1.0,
                            "Wmax": self.W_MAX_TCCX}

        # Connect populations
        nest.Connect(self.cx_pop, self.cx_pop, syn_spec=self.syn_dict_cxcx)         # Cx -> Cx
        nest.Connect(self.cx_pop, self.TC_POP, syn_spec=self.syn_dict_cxtc)         # Cx -> Tc
        nest.Connect(self.TC_POP, self.cx_pop, syn_spec=self.syn_dict_tccx)         # TC -> Cx
    
    def input_contextual_signal(self, neuron_group_id=1):
        """
        Every time a new training image is presented to the network through the thalamic pathway, the facilitation signal 
        coming from the contextual signal provides a 2 kHz Poisson spike train to a different set of 20 neurons, inducing 
        the group to encode for that specific input stimulus.
        
        Turned off during the retrieval phase.
        """
        # Variables
        SET_NEURONS = 20                # Number of population
        contextual_rate = 2000.0        # Hz
        weight_context_cx = 15          # Weight of connection between contextual signal and cx population
        
        # Define set of neurons
        size_group = SET_NEURONS * neuron_group_id
        
        assert (size_group <= self.cx_n), f"Type a value between 1 and {int(self.cx_n/SET_NEURONS)}."
        assert (neuron_group_id > 0), f"Type a value between 1 and {int(self.cx_n/SET_NEURONS)}."
        assert isinstance(neuron_group_id, int), "Type an int value."
        
        # Define slicing
        end_slice =  SET_NEURONS * neuron_group_id
        start_slice = end_slice - SET_NEURONS     
        
        # Generate contextual signal
        self.context_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.context_sign.set(rate=contextual_rate)
        
        # Connect them to the neurons
        nest.Connect(self.context_sign, self.cx_pop[start_slice:end_slice], syn_spec={"weight": weight_context_cx})
        
        # Display connection
        print("Contextual signal successfully connected to the cx population.")           
    
    def input_inhib_signal(self):
        """
        A 10 kHz Poisson spike train is provided to inhibitory neurons to prevent already trained neurons to respond 
        to new stimuli in the training phase.
        
        Only input it after the first training set. Turned off during the retrieval phase.
        """
        # Variables
        inhib_rate = 10000.0           # Hz
        weight_inhib_in = 5            # Weight of inhibitory signal to in population
        
        # Generate inhibitory signal
        self.inhib_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.inhib_sign.set(rate=inhib_rate)
        
        # Connect them to the neurons
        #self.syn_dict = {"weight": weight_inhib_in}
        nest.Connect(self.inhib_sign, self.IN_POP, syn_spec={"weight": weight_inhib_in})   
        
        # Display connection
        print("Inhibitory signal successfully connected to the in population.")     

    def input_train_signal(self):
        """
        During the retrieval phase only the 30 kHz input to thalamic cell is provided, while the contextual signal is off.
        """
        # Variables
        train_rate = 30000.0        # Hz
        weight_train_tc = 5         # Weight of Poisson to tc population
        
        # Generate training signal
        self.train_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.train_sign.set(rate=train_rate)
        
        # Connect them to the neurons
        #self.syn_dict = {"weight": weight_train_tc}
        nest.Connect(self.train_sign, self.TC_POP, syn_spec={"weight": weight_train_tc})
        
        # Display connection
        print("Training signal successfully connected to the tc population.")
        
    def input_sleep(self):
        """
        After the training stage, the sleep-like thalamo-cortical spontaneous slow oscillations activity is induced for 
        a total duration of 600s by providing a non-specific Poisson noise inside the cortex (700 Hz) and increasing the 
        strength of SFA parameter (b=60, in eq. (1)). No external stimulus is provided to cells. Also, the synaptic weights between 
        inhibitory and excitatory neurons in the cortex is reduced to -0.5. In this stage asymmetric STDP plasticity (alpha=3.0) is active in 
        the recurrent cx connectivity, inducing sleep-induced modification in the synaptic weights structure. The parameters’ change to 
        obtain the slow oscillating regime were chosen relying on mean field theory framework18,19.
        """
        # Variables
        osc_rate = 700.0                                    # Hz
        sleep_duration = 600.0                              # ms
        b = 60                                              # SFA parameter
        new_weight_in_cx = -0.5                             # Weight of the synapse between in -> cx population
        alpha_assym = 3.0                                   # Alpha for the assymetric STDP plasticity stage
        params_sleep = {"individual_spike_trains": False}   # False: the generator sends the same spike train to all of its targets
        
        # Change the individual_spike_trains from True (default) to False
        nest.CopyModel("sinusoidal_poisson_generator", "sinusoidal_sleep", params=params_sleep)
        
        # Create sleep oscillation
        self.sleep_osc = nest.Create("sinusoidal_sleep")
        
        # Set frequencies
        self.sleep_osc.set(rate=osc_rate, stop=sleep_duration)
               
        # Set the SFA parameters
        sfa_params = {"b": b}

        # Apply the initial parameters to the population
        nest.SetStatus(self.cx_pop, sfa_params)
        
        # Change connection weight W_in_cx to -0.5 and alpha to 3.0
        # Get Connections
        syn_in_cx = nest.GetConnections(self.IN_POP, self.cx_pop, synapse_model='static_synapse')
        syn_cx_cx = nest.GetConnections(self.cx_pop, self.cx_pop, synapse_model='stdp_synapse')

        # Apply the new parameters to the synapse
        syn_in_cx.set({"weight": new_weight_in_cx})
        syn_cx_cx.set({"alpha": alpha_assym})
        
        # Connect sleep oscillation to the neurons
        nest.Connect(self.sleep_osc, self.cx_pop)
        nest.Connect(self.sleep_osc, self.IN_POP)
        
        # Display connection
        print("Sleep oscillation signal successfully inputed to the cx and in populations (whole cortex).")


