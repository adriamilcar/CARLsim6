#include <carlsim.h>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>

const double PI = 3.14159265358979323846;

// Extract phase of spikes
std::vector< std::vector<double> > phi_distribution(SpikeMonitor* spikeMonitor, double freq) {
    // Calculate the bin time based on frequency
    double bin_time = 1.0 / freq;

    // Vector to store the phase of firing for each neuron
    std::vector< std::vector<double> > phi;

    // Retrieve the spike data from the SpikeMonitor
    const std::vector< std::vector<int> >& spikeTimes = spikeMonitor->getSpikeVector();

    // Iterate over each neuron's spike times
    for (const auto& neuronTimes : spikeTimes) {
        std::vector<double> neuronPhi;
        for (int time : neuronTimes) {
            double timeInSec = time * 1e-3; // Convert time to seconds if needed
            neuronPhi.push_back(2 * PI * std::fmod(timeInSec, bin_time) / bin_time);
        }
        phi.push_back(neuronPhi);
    }

    return phi;
}

int main() {
    const int N = 6; // Number of populations
    const int M = 100; // Number of neurons per population

    // CARLsim simulator instance
    CARLsim sim("CA1", CPU_MODE, USER);

    // LIF neuron parameters
    float R_m = 142e6;    // Membrane resistance in ohms
    float tau_m = 24e-3;  // Membrane time constant in seconds
    float v_rest = -65e-3; // Resting potential in volts
    float v_thres = -50e-3; // Threshold potential in volts
    float v_reset = -65e-3; // Reset potential in volts
    float tau_ref = 5e-3;  // Refractory period in seconds

    // Create neuron groups
    std::vector<int> groups(N);
    std::vector<SpikeMonitor*> spikeMonitors(N);
    std::vector<NeuronMonitor*> neuronMonitors(N);
    std::vector<ConnectionMonitor*> connMonitors;
    for (int i = 0; i < N; ++i) {
        // Create groups of LIF neurons
        groups[i] = sim.createGroup("LIF_neurons_" + std::to_string(i), M, EXCITATORY_NEURON);
        sim.setNeuronParametersLIF(groups[i], v_rest, v_reset, tau_m, R_m, v_thres, tau_ref);

        // Set up spike and neuron monitors for each group
        spikeMonitors[i] = sim.setSpikeMonitor(groups[i], "DEFAULT");
        neuronMonitors[i] = sim.setNeuronMonitor(groups[i], "DEFAULT");
    }

    // Define synaptic weights and setup connections
    float g_inh_weight = 0.2f; // Weight for global inhibition
    float g_ahp_weight = 0.3f; // Weight for AHP
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Setup global inhibition
            sim.connect(groups[i], groups[j], "full", RangeWeight(g_inh_weight), 1.0f);
            // Setup AHP (self-connection)
            if (i == j) {
                sim.connect(groups[i], groups[j], "one-to-one", RangeWeight(g_ahp_weight), 1.0f);
            }
            // Setup plastic connections with STDP between different populations
            if (i != j) {
                sim.connect(groups[i], groups[j], "random", RangeWeight(0.0, 1.0f/100, 20.0f/100), 
                            1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);

                // STDP parameters
                float ALPHA_LTP = 0.005f; // LTP rate constant
                float TAU_LTP = 20.0f;    // Time constant for LTP
                float ALPHA_LTD = 0.005f; // LTD rate constant
                float TAU_LTD = 20.0f;    // Time constant for LTD

                // Set E-STDP parameters
                sim.setESTDP(groups[i], groups[j], true, STANDARD, 
                             ExpCurve(ALPHA_LTP, TAU_LTP, ALPHA_LTD, TAU_LTD));

                // Set up a ConnectionMonitor for each plastic connection
                connMonitors.push_back(sim.setConnectionMonitor(groups[i], groups[j], "DEFAULT"));
            }
        }
    }

    // Oscillation and noise parameters
    float I_osc = 40e-12;  // Oscillation amplitude in picoamps
    float oscillation_T = 200e-3;  // Oscillation period in seconds
    float f = 1 / oscillation_T;   // Frequency in Hz

    float noise_frac = 0.04;    // Noise strenght (eta)
    std::default_random_engine generator;
    std::normal_distribution<float> noise_distribution(0.0, sqrt(noise_frac * (v_thres - v_rest) / tau_m));

    // Setup network
    sim.setupNetwork();
    sim.saveSimulation("initialNetworkState.dat", true);

    // Start recording
    for (int i = 0; i < N; ++i) {
        spikeMonitors[i]->startRecording();
        neuronMonitors[i]->startRecording();
    }

    // Stimulus currents
    std::vector<float> I_s(N);  
    for (int i = 0; i < N; ++i) {
        I_s[i] = (120 + i * (20.0 / N)) * 1e-12; // Linearly spaced tonic currents
    }

    // Run the simulation
    for (int t = 0; t < 1000; ++t) {
        for (int i = 0; i < N; ++i) {
            std::vector<float> current(M, I_s[i]);
            for (int j = 0; j < M; ++j) {
                float I_osc_current = I_osc * cos(2 * M_PI * f * t * 1e-3 - M_PI);
                float noise_current = noise_distribution(generator) * 1e-12;
                current[j] += I_osc_current + noise_current;
            }
            sim.setExternalCurrent(groups[i], current);
        }
        sim.runNetwork(0, 1); // Run for 1ms
    }

    // Stop recording and process data
    for (int i = 0; i < N; ++i) {
        spikeMonitors[i]->stopRecording();
        neuronMonitors[i]->stopRecording();
        
        spikeMonitors[i]->print();
        neuronMonitors[i]->print();
    }

    // Take snapshots of the weights and print them
    for (auto& cm : connMonitors) {
        cm->takeSnapshot();
        cm->print();
    }

    // Save final network state
    sim.saveSimulation("finalNetworkState.dat", true);

    // Output the phase data to a file
    std::vector< std::vector<double> > spikePhases = phi_distribution(spikeMonitors[0], f); // For first group
    std::ofstream outFile("phaseData.csv");
    for (const auto& neuronPhases : spikePhases) {
        for (double phase : neuronPhases) {
            outFile << phase << ",";
        }
        outFile << "\n";
    }
    outFile.close();

    return 0;
}


