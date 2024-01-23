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
        self.cx_pop = nest.Create('aeif_cond_alpha', self.cx_n)
        self.in_pop = nest.Create('aeif_cond_alpha', self.IN_N)
        self.tc_pop = nest.Create('aeif_cond_alpha', self.TC_N)
        self.re_pop = nest.Create('aeif_cond_alpha', self.RE_N)
        
        # Connect populations with static synapses
        in_cx = nest.Connect(self.in_pop, self.cx_pop, syn_spec={"weight": -4}) # inhibitory interneurons -> pyramidal neurons
        cx_in = nest.Connect(self.cx_pop, self.in_pop, syn_spec={"weight": 60}) # pyramidal neurons -> inhibitory interneurons
        tc_re = nest.Connect(self.tc_pop, self.re_pop, syn_spec={"weight": 10}) # thalamic relay -> reticular neurons
        re_tc = nest.Connect(self.re_pop, self.tc_pop, syn_spec={"weight": -10}) # reticular neurons -> thalamic relay
        in_in = nest.Connect(self.in_pop, self.in_pop, syn_spec={"weight": -1}) # inhibitory interneurons -> inhibitory interneuros
        re_re = nest.Connect(self.re_pop, self.re_pop, syn_spec={"weight": -1}) # reticular neurons -> reticular neurons
        
        # Connect populations with STDP synapses
        self.W_MAX_CXCX = 15                        # Max weight value fo the cx-cx connection: 150. I've changed to 15 to make it run for the tests.     
        self.W_MAX_CXTC = 13                        # Max weight value fo the cx-tc connection: 130 I've changed to 13 to make it run for the tests.
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
        cx_cx = nest.Connect(self.cx_pop, self.cx_pop, syn_spec=self.syn_dict_cxcx)
        cx_tc = nest.Connect(self.cx_pop, self.tc_pop, syn_spec=self.syn_dict_cxtc)
        tc_cx = nest.Connect(self.tc_pop, self.cx_pop, syn_spec=self.syn_dict_tccx)
    
    def input_contextual_signal(self):
        # Contextual signal (Poisson spile train of 2 kHz) to the cx population
        # Turn off during the retrieval phase
        self.context_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.context_sign.set(rate=2000.0)
        
        # Connect them to the neurons
        self.syn_dict = {"weight": 15}
        nest.Connect(self.context_sign, self.cx_pop, syn_spec=self.syn_dict)
        
        # Display connection
        print("Contextual signal successfully connected to the cx population.")           
    
    def input_inhib_signal(self):
        # Poisson spike train to inhibitory neurons of 10 kHz
        # Only connect after the first training set. Turn off during the retrieval phase.
        self.inhib_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.inhib_sign.set(rate=10000.0)
        
        # Connect them to the neurons
        self.syn_dict = {"weight": 5}
        nest.Connect(self.inhib_sign, self.in_pop, syn_spec=self.syn_dict)   
        
        # Display connection
        print("Inhibitory signal successfully connected to the in population.")     

    def input_train_signal(self):
        # Only connect after the first training set. Turn off during the retrieval phase.
        self.train_sign = nest.Create("poisson_generator")
        
        # Set frequencies
        self.train_sign.set(rate=30000.0)
        
        # Connect them to the neurons
        self.syn_dict = {"weight": 5}
        nest.Connect(self.train_sign, self.tc_pop, syn_spec=self.syn_dict)
        
        # Display connection
        print("Training signal successfully connected to the tc population.")
        
    def input_sleep(self):
        """
            After the training stage, the sleep-like thalamo-cortical spontaneous slow oscillations activity is induced for 
            a total duration of 600s by providing a non-specific Poisson noise inside the cortex (700 Hz) and increasing the 
            strength of SFA parameter (, in eq. (1)). No external stimulus is provided to cells. Also, the synaptic weights between 
            inhibitory and excitatory neurons in the cortex is reduced to . In this stage asymmetric STDP plasticity () is active in 
            the recurrent connectivity, inducing sleep-induced modification in the synaptic weights structure. The parameters’ change to 
            obtain the slow oscillating regime were chosen relying on mean field theory framework18,19.
        """
        pass


