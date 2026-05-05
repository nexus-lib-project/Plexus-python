% neurochemical_simulation.m - MATLAB Neurochemical Brain Simulation
% Version 1.0: Advanced Neurochemical Dynamics for AI Emotion System
%
% This MATLAB script provides comprehensive neurochemical simulations
% for the AI emotion system, including:
% - Neurotransmitter dynamics
% - Brain region activity modeling
% - Neural firing pattern simulation
% - Emotional state computation
%
% Usage from Python:
%   import matlab.engine
%   eng = matlab.engine.start_matlab()
%   result = eng.run_neurochemical_simulation(inputs)

classdef neurochemical_simulation
    properties
        % Simulation parameters
        time_step = 0.001;  % seconds
        duration = 1.0;     % seconds
        
        % Neurochemical baseline levels
        neurochemical_baselines = struct(...
            'dopamine', 0.5, ...
            'serotonin', 0.5, ...
            'norepinephrine', 0.4, ...
            'GABA', 0.6, ...
            'glutamate', 0.5, ...
            'acetylcholine', 0.5, ...
            'oxytocin', 0.3, ...
            'endorphins', 0.2, ...
            'anandamide', 0.2, ...
            'CRF', 0.3, ...
            'histamine', 0.4, ...
            'adenosine', 0.3, ...
            'melatonin', 0.2, ...
            'substance_p', 0.2, ...
            'neuropeptide_y', 0.3, ...
            'vasopressin', 0.3, ...
            'testosterone', 0.4, ...
            'estrogen', 0.4, ...
            'insulin', 0.5);
        
        % Brain region baseline activity
        brain_region_baselines = struct(...
            'amygdala', 0.3, ...
            'nucleus_accumbens', 0.3, ...
            'ventral_tegmental_area', 0.3, ...
            'raphe_nuclei', 0.3, ...
            'locus_coeruleus', 0.3, ...
            'prefrontal_cortex', 0.5, ...
            'hippocampus', 0.4, ...
            'hypothalamus', 0.4, ...
            'anterior_cingulate', 0.4, ...
            'insula', 0.3, ...
            'orbitofrontal_cortex', 0.4);
        
        % Current state
        current_neurochemicals
        current_brain_regions
        current_time
    end
    
    methods
        function obj = neurochemical_simulation()
            % Initialize current state from baselines
            obj.current_neurochemicals = obj.neurochemical_baselines;
            obj.current_brain_regions = obj.brain_region_baselines;
            obj.current_time = 0;
        end
        
        % Main simulation function
        function result = run_simulation(obj, input_signal, duration)
            % Run neurochemical simulation
            % input_signal: struct with emotional/sensory input
            % duration: simulation duration in seconds
            
            if nargin < 3
                duration = obj.duration;
            end
            
            % Time vector
            t = 0:obj.time_step:duration;
            n_steps = length(t);
            
            % Initialize state history
            neurochem_history = struct();
            regions_history = struct();
            
            for field = fieldnames(obj.current_neurochemicals)'
                neurochem_history.(field{1}) = zeros(1, n_steps);
            end
            
            for field = fieldnames(obj.current_brain_regions)'
                regions_history.(field{1}) = zeros(1, n_steps);
            end
            
            % Run simulation
            for i = 1:n_steps
                obj.current_time = t(i);
                
                % Update neurochemicals based on input
                obj = obj.update_neurochemicals(input_signal, obj.time_step);
                
                % Update brain regions based on neurochemicals
                obj = obj.update_brain_regions(obj.time_step);
                
                % Record state
                for field = fieldnames(obj.current_neurochemicals)'
                    neurochem_history.(field{1})(i) = obj.current_neurochemicals.(field{1});
                end
                
                for field = fieldnames(obj.current_brain_regions)'
                    regions_history.(field{1})(i) = obj.current_brain_regions.(field{1});
                end
            end
            
            % Compile results
            result = struct();
            result.time = t;
            result.neurochemicals = neurochem_history;
            result.brain_regions = regions_history;
            result.final_neurochemicals = obj.current_neurochemicals;
            result.final_brain_regions = obj.current_brain_regions;
            result.emotional_state = obj.compute_emotional_state();
            result.dominant_firing_pattern = obj.compute_firing_pattern();
        end
        
        % Update neurochemical levels
        function obj = update_neurochemicals(obj, input_signal, dt)
            % Neurotransmitter dynamics with input effects
            
            % Get input valence and arousal
            valence = 0.5;
            arousal = 0.5;
            stress = 0.0;
            
            if isfield(input_signal, 'valence')
                valence = input_signal.valence;
            end
            if isfield(input_signal, 'arousal')
                arousal = input_signal.arousal;
            end
            if isfield(input_signal, 'stress')
                stress = input_signal.stress;
            end
            
            % Dopamine: reward prediction error
            dopamine_input = max(0, valence - 0.5) * arousal;
            obj.current_neurochemicals.dopamine = obj.update_concentration(...
                obj.current_neurochemicals.dopamine, ...
                obj.neurochemical_baselines.dopamine, ...
                dopamine_input, 0.1, 0.05, dt);
            
            % Serotonin: mood regulation, affected by positive valence
            serotonin_input = valence * 0.3;
            obj.current_neurochemicals.serotonin = obj.update_concentration(...
                obj.current_neurochemicals.serotonin, ...
                obj.neurochemical_baselines.serotonin, ...
                serotonin_input, 0.08, 0.04, dt);
            
            % Norepinephrine: arousal and attention
            norepinephrine_input = arousal * 0.5;
            obj.current_neurochemicals.norepinephrine = obj.update_concentration(...
                obj.current_neurochemicals.norepinephrine, ...
                obj.neurochemical_baselines.norepinephrine, ...
                norepinephrine_input, 0.12, 0.06, dt);
            
            % GABA: inhibitory, increases with low arousal
            gaba_input = (1 - arousal) * 0.3;
            obj.current_neurochemicals.GABA = obj.update_concentration(...
                obj.current_neurochemicals.GABA, ...
                obj.neurochemical_baselines.GABA, ...
                gaba_input, 0.07, 0.03, dt);
            
            % Glutamate: excitatory, increases with arousal
            glutamate_input = arousal * 0.4;
            obj.current_neurochemicals.glutamate = obj.update_concentration(...
                obj.current_neurochemicals.glutamate, ...
                obj.neurochemical_baselines.glutamate, ...
                glutamate_input, 0.1, 0.05, dt);
            
            % Acetylcholine: attention and learning
            ach_input = arousal * 0.3;
            obj.current_neurochemicals.acetylcholine = obj.update_concentration(...
                obj.current_neurochemicals.acetylcholine, ...
                obj.neurochemical_baselines.acetylcholine, ...
                ach_input, 0.09, 0.04, dt);
            
            % Oxytocin: social bonding, positive valence
            oxytocin_input = max(0, valence - 0.3) * 0.4;
            obj.current_neurochemicals.oxytocin = obj.update_concentration(...
                obj.current_neurochemicals.oxytocin, ...
                obj.neurochemical_baselines.oxytocin, ...
                oxytocin_input, 0.05, 0.02, dt);
            
            % Endorphins: pleasure and reward
            endorphin_input = max(0, valence - 0.5) * arousal * 0.5;
            obj.current_neurochemicals.endorphins = obj.update_concentration(...
                obj.current_neurochemicals.endorphins, ...
                obj.neurochemical_baselines.endorphins, ...
                endorphin_input, 0.06, 0.03, dt);
            
            % CRF: stress response
            crf_input = stress * 0.6;
            obj.current_neurochemicals.CRF = obj.update_concentration(...
                obj.current_neurochemicals.CRF, ...
                obj.neurochemical_baselines.CRF, ...
                crf_input, 0.15, 0.08, dt);
            
            % Adenosine: fatigue (accumulates over time)
            adenosine_input = 0.01;  % Baseline accumulation
            obj.current_neurochemicals.adenosine = obj.update_concentration(...
                obj.current_neurochemicals.adenosine, ...
                obj.neurochemical_baselines.adenosine, ...
                adenosine_input, 0.02, 0.01, dt);
        end
        
        % Update single neurochemical concentration
        function new_conc = update_concentration(obj, current, baseline, input_val, synthesis_rate, degradation_rate, dt)
            % Simple ODE: dC/dt = synthesis*(baseline - C) - degradation*C + input
            dC = synthesis_rate * (baseline - current) - degradation_rate * current + input_val;
            new_conc = current + dC * dt;
            new_conc = max(0, min(1, new_conc));  % Clamp to [0, 1]
        end
        
        % Update brain region activity
        function obj = update_brain_regions(obj, dt)
            % Amygdala: fear and emotion processing
            amygdala_input = obj.current_neurochemicals.CRF * 0.4 + ...
                            obj.current_neurochemicals.norepinephrine * 0.3;
            obj.current_brain_regions.amygdala = obj.update_region(...
                obj.current_brain_regions.amygdala, ...
                obj.brain_region_baselines.amygdala, ...
                amygdala_input, dt);
            
            % Nucleus accumbens: reward processing
            nacc_input = obj.current_neurochemicals.dopamine * 0.5 + ...
                        obj.current_neurochemicals.endorphins * 0.3;
            obj.current_brain_regions.nucleus_accumbens = obj.update_region(...
                obj.current_brain_regions.nucleus_accumbens, ...
                obj.brain_region_baselines.nucleus_accumbens, ...
                nacc_input, dt);
            
            % Ventral tegmental area: dopamine production
            vta_input = obj.current_neurochemicals.glutamate * 0.4 - ...
                       obj.current_neurochemicals.GABA * 0.3;
            obj.current_brain_regions.ventral_tegmental_area = obj.update_region(...
                obj.current_brain_regions.ventral_tegmental_area, ...
                obj.brain_region_baselines.ventral_tegmental_area, ...
                vta_input, dt);
            
            % Raphe nuclei: serotonin production
            raphe_input = 0.2;  % Baseline activity
            obj.current_brain_regions.raphe_nuclei = obj.update_region(...
                obj.current_brain_regions.raphe_nuclei, ...
                obj.brain_region_baselines.raphe_nuclei, ...
                raphe_input, dt);
            
            % Locus coeruleus: norepinephrine production
            lc_input = obj.current_neurochemicals.CRF * 0.5 + ...
                      obj.current_neurochemicals.acetylcholine * 0.2;
            obj.current_brain_regions.locus_coeruleus = obj.update_region(...
                obj.current_brain_regions.locus_coeruleus, ...
                obj.brain_region_baselines.locus_coeruleus, ...
                lc_input, dt);
            
            % Prefrontal cortex: executive function
            pfc_input = obj.current_neurochemicals.dopamine * 0.3 + ...
                       obj.current_neurochemicals.norepinephrine * 0.2 + ...
                       obj.current_neurochemicals.acetylcholine * 0.2 - ...
                       obj.current_neurochemicals.CRF * 0.2;
            obj.current_brain_regions.prefrontal_cortex = obj.update_region(...
                obj.current_brain_regions.prefrontal_cortex, ...
                obj.brain_region_baselines.prefrontal_cortex, ...
                pfc_input, dt);
            
            % Hippocampus: memory
            hippo_input = obj.current_neurochemicals.serotonin * 0.3 + ...
                         obj.current_neurochemicals.acetylcholine * 0.3 - ...
                         obj.current_neurochemicals.CRF * 0.3;
            obj.current_brain_regions.hippocampus = obj.update_region(...
                obj.current_brain_regions.hippocampus, ...
                obj.brain_region_baselines.hippocampus, ...
                hippo_input, dt);
            
            % Hypothalamus: homeostasis
            hypo_input = obj.current_neurochemicals.CRF * 0.3 + ...
                        obj.current_neurochemicals.adenosine * 0.2;
            obj.current_brain_regions.hypothalamus = obj.update_region(...
                obj.current_brain_regions.hypothalamus, ...
                obj.brain_region_baselines.hypothalamus, ...
                hypo_input, dt);
            
            % Anterior cingulate: conflict monitoring
            acc_input = obj.current_neurochemicals.norepinephrine * 0.3 + ...
                       obj.current_neurochemicals.CRF * 0.2;
            obj.current_brain_regions.anterior_cingulate = obj.update_region(...
                obj.current_brain_regions.anterior_cingulate, ...
                obj.brain_region_baselines.anterior_cingulate, ...
                acc_input, dt);
            
            % Insula: interoception
            insula_input = obj.current_neurochemicals.CRF * 0.3 + ...
                          obj.current_neurochemicals.substance_p * 0.2;
            obj.current_brain_regions.insula = obj.update_region(...
                obj.current_brain_regions.insula, ...
                obj.brain_region_baselines.insula, ...
                insula_input, dt);
            
            % Orbitofrontal cortex: value representation
            ofc_input = obj.current_neurochemicals.dopamine * 0.3 + ...
                       obj.current_neurochemicals.serotonin * 0.2;
            obj.current_brain_regions.orbitofrontal_cortex = obj.update_region(...
                obj.current_brain_regions.orbitofrontal_cortex, ...
                obj.brain_region_baselines.orbitofrontal_cortex, ...
                ofc_input, dt);
        end
        
        % Update single brain region activity
        function new_activity = update_region(obj, current, baseline, input_val, dt)
            tau = 0.1;  % Time constant
            dA = (baseline + input_val - current) / tau;
            new_activity = current + dA * dt;
            new_activity = max(0, min(1, new_activity));  % Clamp to [0, 1]
        end
        
        % Compute emotional state from neurochemicals
        function emotional_state = compute_emotional_state(obj)
            emotional_state = struct();
            
            % Primary emotions based on neurochemical patterns
            dopamine = obj.current_neurochemicals.dopamine;
            serotonin = obj.current_neurochemicals.serotonin;
            norepinephrine = obj.current_neurochemicals.norepinephrine;
            crf = obj.current_neurochemicals.CRF;
            oxytocin = obj.current_neurochemicals.oxytocin;
            endorphins = obj.current_neurochemicals.endorphins;
            
            % Joy: high dopamine, high serotonin, low CRF
            emotional_state.joy = dopamine * serotonin * (1 - crf);
            
            % Fear: high CRF, high norepinephrine, low serotonin
            emotional_state.fear = crf * norepinephrine * (1 - serotonin);
            
            % Anger: high CRF, high norepinephrine, low serotonin, low oxytocin
            emotional_state.anger = crf * norepinephrine * (1 - serotonin) * (1 - oxytocin);
            
            % Sadness: low dopamine, low serotonin, moderate CRF
            emotional_state.sadness = (1 - dopamine) * (1 - serotonin) * crf;
            
            % Surprise: high norepinephrine, moderate dopamine
            emotional_state.surprise = norepinephrine * (0.5 + dopamine * 0.5);
            
            % Disgust: high CRF, low oxytocin
            emotional_state.disgust = crf * (1 - oxytocin);
            
            % Trust: high oxytocin, high serotonin, low CRF
            emotional_state.trust = oxytocin * serotonin * (1 - crf);
            
            % Anticipation: high dopamine, high norepinephrine
            emotional_state.anticipation = dopamine * norepinephrine;
            
            % Love: high oxytocin, high dopamine, high serotonin
            emotional_state.love = oxytocin * dopamine * serotonin;
            
            % Contentment: high serotonin, moderate dopamine, low CRF
            emotional_state.contentment = serotonin * (0.5 + dopamine * 0.5) * (1 - crf);
            
            % Find dominant emotion
            emotions = fieldnames(emotional_state);
            max_val = 0;
            dominant_emotion = 'neutral';
            for i = 1:length(emotions)
                val = emotional_state.(emotions{i});
                if val > max_val
                    max_val = val;
                    dominant_emotion = emotions{i};
                end
            end
            
            emotional_state.dominant = dominant_emotion;
            emotional_state.dominant_intensity = max_val;
        end
        
        % Compute neural firing pattern
        function pattern = compute_firing_pattern(obj)
            dopamine = obj.current_neurochemicals.dopamine;
            norepinephrine = obj.current_neurochemicals.norepinephrine;
            gaba = obj.current_neurochemicals.GABA;
            
            % Determine dominant firing pattern
            if norepinephrine > 0.7 && dopamine > 0.6
                pattern = 'burst';
            elseif gaba > 0.7
                pattern = 'tonic';
            elseif dopamine > 0.6
                pattern = 'phasic';
            else
                pattern = 'tonic';
            end
        end
        
        % Compute brain waves
        function brain_waves = compute_brain_waves(obj, signal, fs)
            % Compute power in different frequency bands
            if nargin < 3
                fs = 1000;  % Default sampling rate
            end
            
            % FFT
            n = length(signal);
            fft_signal = fft(signal);
            power = abs(fft_signal(1:floor(n/2))).^2;
            freqs = (0:floor(n/2)-1) * fs / n;
            
            % Frequency bands
            brain_waves = struct();
            
            % Delta (0.5-4 Hz)
            delta_idx = freqs >= 0.5 & freqs < 4;
            brain_waves.delta = mean(power(delta_idx));
            
            % Theta (4-8 Hz)
            theta_idx = freqs >= 4 & freqs < 8;
            brain_waves.theta = mean(power(theta_idx));
            
            % Alpha (8-13 Hz)
            alpha_idx = freqs >= 8 & freqs < 13;
            brain_waves.alpha = mean(power(alpha_idx));
            
            % Beta (13-30 Hz)
            beta_idx = freqs >= 13 & freqs < 30;
            brain_waves.beta = mean(power(beta_idx));
            
            % Gamma (30-100 Hz)
            gamma_idx = freqs >= 30 & freqs < 100;
            brain_waves.gamma = mean(power(gamma_idx));
        end
        
        % Inject neurochemical (for external modulation)
        function obj = inject_neurochemical(obj, name, amount)
            if isfield(obj.current_neurochemicals, name)
                obj.current_neurochemicals.(name) = ...
                    max(0, min(1, obj.current_neurochemicals.(name) + amount));
            end
        end
        
        % Get current state as struct
        function state = get_state(obj)
            state = struct();
            state.neurochemicals = obj.current_neurochemicals;
            state.brain_regions = obj.current_brain_regions;
            state.emotional_state = obj.compute_emotional_state();
            state.firing_pattern = obj.compute_firing_pattern();
            state.time = obj.current_time;
        end
        
        % Reset to baseline
        function obj = reset(obj)
            obj.current_neurochemicals = obj.neurochemical_baselines;
            obj.current_brain_regions = obj.brain_region_baselines;
            obj.current_time = 0;
        end
    end
    
    % Static methods for standalone function calls
    methods (Static)
        % Quick simulation function for Python interface
        function result = quick_simulate(valence, arousal, stress, duration)
            if nargin < 4
                duration = 1.0;
            end
            if nargin < 3
                stress = 0.0;
            end
            if nargin < 2
                arousal = 0.5;
            end
            if nargin < 1
                valence = 0.5;
            end
            
            sim = neurochemical_simulation();
            input_signal = struct('valence', valence, 'arousal', arousal, 'stress', stress);
            result = sim.run_simulation(input_signal, duration);
        end
        
        % Analyze text sentiment and simulate
        function result = analyze_text(text)
            % Simple sentiment analysis
            positive_words = {'happy', 'joy', 'love', 'great', 'good', 'wonderful', 'amazing', 'beautiful'};
            negative_words = {'sad', 'hate', 'angry', 'bad', 'terrible', 'awful', 'horrible', 'fear'};
            arousal_words = {'excited', 'amazing', 'terrible', 'intense', 'extreme', 'urgent'};
            
            text_lower = lower(text);
            
            valence = 0.5;
            arousal = 0.5;
            stress = 0.0;
            
            for i = 1:length(positive_words)
                if contains(text_lower, positive_words{i})
                    valence = valence + 0.1;
                end
            end
            
            for i = 1:length(negative_words)
                if contains(text_lower, negative_words{i})
                    valence = valence - 0.1;
                    stress = stress + 0.1;
                end
            end
            
            for i = 1:length(arousal_words)
                if contains(text_lower, arousal_words{i})
                    arousal = arousal + 0.1;
                end
            end
            
            valence = max(0, min(1, valence));
            arousal = max(0, min(1, arousal));
            stress = max(0, min(1, stress));
            
            sim = neurochemical_simulation();
            input_signal = struct('valence', valence, 'arousal', arousal, 'stress', stress);
            result = sim.run_simulation(input_signal, 1.0);
            result.input_valence = valence;
            result.input_arousal = arousal;
            result.input_stress = stress;
        end
    end
end

% Standalone function interface for Python
function result = run_neurochemical_simulation(input_struct)
% RUN_NEUROCHEMICAL_SIMULATION - Main entry point for Python interface
%
% input_struct fields:
%   - valence: 0-1 (negative to positive)
%   - arousal: 0-1 (calm to excited)
%   - stress: 0-1 (relaxed to stressed)
%   - duration: simulation duration in seconds (optional)
%
% Returns:
%   - final_neurochemicals: struct with final neurochemical levels
%   - final_brain_regions: struct with final brain region activity
%   - emotional_state: struct with computed emotions
%   - dominant_firing_pattern: string ('tonic', 'phasic', 'burst')

    sim = neurochemical_simulation();
    
    valence = 0.5;
    arousal = 0.5;
    stress = 0.0;
    duration = 1.0;
    
    if isfield(input_struct, 'valence')
        valence = input_struct.valence;
    end
    if isfield(input_struct, 'arousal')
        arousal = input_struct.arousal;
    end
    if isfield(input_struct, 'stress')
        stress = input_struct.stress;
    end
    if isfield(input_struct, 'duration')
        duration = input_struct.duration;
    end
    
    input_signal = struct('valence', valence, 'arousal', arousal, 'stress', stress);
    result = sim.run_simulation(input_signal, duration);
end
