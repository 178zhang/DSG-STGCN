function [EEG_data_final, source_all] = generate_data_region_all_neighbors_all_AR_all(leadfield, neighbors_info, regions_index, pm, regions, neighbors)


%% set up default values and read parameters.
if isfield(pm,'recording_time')
    recording_time = pm.recording_time;
else 
    recording_time = 1;
end

if isfield(pm,'frequency')
    fz = pm.frequency;
else 
    fz = 100; 
end

if isfield(pm, 'use_AR')
    use_AR = pm.use_AR;
else % defaulse: false
    use_AR = false;
end

if isfield(pm,'addSensorNoise')
    addSensorNoise = pm.addSensorNoise;
    if addSensorNoise == true && isfield(pm, 'SNR_sensor')
        SNR_sensor = pm.SNR_sensor;
    elseif addSensorNoise == true && ~isfield(pm, 'SNR_sensor')
        error('must specify sensor SNR level!!')
    elseif addSensorNoise == false
          SNR_sensor = 10000;      
    end    
else % defaulse: not adding any sensor noise.
    addSensorNoise = false;
    SNR_sensor = 10000;    
end

if isfield(pm,'addSourceSpaceNoise')
    addSourceSpaceNoise = pm.addSourceSpaceNoise;
    if addSourceSpaceNoise == true
        if isfield(pm, 'noise_type')
            noise_type = pm.noise_type;  % Guassian or Spike.
        else
            error('must specify noise type in the source space!!')
        end

        if isfield(pm, 'SNR_source')
            SNR_source = pm.SNR_source;
        else
            error('must specify source SNR level!!')
        end

        if isfield(pm, 'spike_number') % only use when using spike noise.
            spike_number = pm.spike_number;
        else
            spike_number = 3;
        end
    else % specified not adding 
        SNR_source = 10000;
        noise_type = 'NA';
        spike_number = 0;
    end
    
else % if no addSourceSpaceNoise in the field; By default.
        addSourceSpaceNoise = false;
        SNR_source = 10000;
        noise_type = 'NA';
        spike_number = 0;    
end

%% find source locations.
% regions_index: 1*2 [index_1, index_2]
Nt = recording_time*fz;
actual_source = zeros(size(leadfield,2), Nt);
% source_locations = idx';% 2*1

% set neighbors.
if neighbors==1
    source_neighbors = neighbors_info.src_1_neighbors(:, regions_index);
elseif neighbors==2
    source_neighbors = neighbors_info.src_2_neighbors(:, regions_index);
elseif neighbors==3
    source_neighbors = neighbors_info.src_3_neighbors(:, regions_index);
elseif neighbors==4
    source_neighbors = neighbors_info.src_4_neighbors(:, regions_index);
else % neighbors==5
    source_neighbors = neighbors_info.src_5_neighbors(:, regions_index);
end
source_neighbors = full(source_neighbors);
source_neighbors = single(source_neighbors);

% fill the time series to source locations.
if use_AR == true
    if regions == 1
        regions_temp = 2;
        [~, sources_nonint, ~] = generate_sources_ar(fz, Nt/fz , [1 40], regions_temp);
        sources_nonint = sources_nonint(1,:);
    else % region > 1
        [~, sources_nonint, ~] = generate_sources_ar(fz, Nt/fz , [1 40], regions);
    end

    sources_signal_activated = sources_nonint./repmat(std(sources_nonint,0, 2),[1,size(sources_nonint,2)]);% [1,50] row*column


else % use_AR == false
    sources_signal_activated = ones([regions, Nt]);
end

for i = 1:regions
    ar_strength_temp = repmat(sources_signal_activated(i,:),[size(leadfield,2),1]);% [2052,1] row*column
    source_neighbors_temp = repmat(source_neighbors(:,i),[1,Nt]);
    actual_source_temp = source_neighbors_temp.*ar_strength_temp;
    actual_source = actual_source + actual_source_temp;
end

% ===================== end =========================

signal_energy = norm(actual_source, 'fro')^2;

if addSourceSpaceNoise == true
    switch noise_type
        case 'Guassian'
            % class 1
            source_noise =randn(size(actual_source));
            noise_power = signal_energy/(10^(SNR_source/10));
            source_noise = source_noise/(norm(source_noise,'fro'))*sqrt(noise_power);
            
        case 'Spike'
            % class 1
            source_noise = get_spike_noise_matrix(spike_number, actual_source);
            noise_power = signal_energy/(10^(SNR_source/10));
            source_noise = source_noise/(norm(source_noise,'fro'))*sqrt(noise_power);
    end
else % no source noise.
    source_noise = zeros(size(actual_source));
end


source_all = actual_source + source_noise;

% true_source.source = actual_source;
% true_source.source_noise = source_noise;
% true_source.source_all = source_all;
% 
% true_source.description = 'source_all = source + source_noise ; EEG_signal_ = L_2k * source_all;';

EEG_signal = leadfield * source_all;


if addSensorNoise == 0 % No sensor level noise!
    EEG_data_final = EEG_signal;
else
    %% YL rewrite how to add noise
    % EEG_signal = [EEG_signal, EEG_signal_2 EEG_signal_3];
    EEG_energy = norm(EEG_signal, 'fro')^2;
    EEG_noise =randn(size(EEG_signal));
    
    noise_power_sensor = EEG_energy/(10^(SNR_sensor/10));
    EEG_noise = EEG_noise/(norm(EEG_noise,'fro'))*sqrt(noise_power_sensor);
    EEG_data_final = EEG_signal + EEG_noise;
    
    %EEG_data_1 = EEG_signal_all(:,1:NN);
    %EEG_data_2 = EEG_signal_all(:,NN+1:NN*2);
    %EEG_data_3 = EEG_signal_all(:,2*NN+1:end);
end


% EEG_data.data = EEG_data_final;
% 
% EEG_data.description  = strcat( 'addSourceSpaceNoise_', num2str(addSourceSpaceNoise), '_addSensorNoise_', num2str(addSensorNoise), ...
%     '_SNR_source_', num2str(SNR_source), '_SNR_sensor_', num2str(SNR_sensor), '_useAR_', num2str(use_AR), '_noisetype_', noise_type, ...
%     '_spike_number_', num2str(spike_number));
% 
% EEG_data.true_source = true_source;

end