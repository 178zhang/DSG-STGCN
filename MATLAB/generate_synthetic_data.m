%% 2052: generate synthetic source & EEG data
% 20 regions

clear all
close all
clc

% addpath('FISTA-master')
% addpath('FISTA-master/proj')
% addpath('FISTA-master/utils')
addpath(genpath('BBCBtools'))
% addpath('NeuroIP')
% addpath('mgh_plotting')
% addpath('utils')
addpath(genpath('mne_matlab'))
% addpath('MMT-master')

% root_path = 'C:\MyData\Local\20221224\generate_brain_data';
root_path = 'generate_brain_data';
subject_path = [root_path, '\2052_files'];
addpath(root_path)
addpath(subject_path)


%% loading data
data_save_path = 'Brain_dataset\';
if ~exist(data_save_path, 'dir')
    mkdir(data_save_path)
end

shortest_dist_mat = [subject_path, '\G_short_dist_mat.mat'];
leadfiled_mat = [subject_path, '\new_leadfield.mat'];

src_1 = load('src_1.mat');
src_1 = src_1.src_1;
src_2 = load('src_2.mat');
src_2 = src_2.src_2;

fwd.src = [];
fwd.src = [fwd.src, src_1];
fwd.src = [fwd.src, src_2];
    
if isfile(shortest_dist_mat) % 0
    load(shortest_dist_mat, 'src_graph_binary', 'G_short_dist_mat')
else
    [src_graph_binary, src_graph_weighted, G_short_dist_mat] = get_source_neighbor_matrix(fwd.src);
    save(shortest_dist_mat, 'src_graph_binary', 'src_graph_weighted', 'G_short_dist_mat')
end

% Activation part 
src_1_neighbors = get_src_neighbors(src_graph_binary, 1);
src_2_neighbors = get_src_neighbors(src_graph_binary, 2);
src_3_neighbors = get_src_neighbors(src_graph_binary, 3);

%
only_3 = src_3_neighbors-src_2_neighbors;
only_2 = src_2_neighbors-src_1_neighbors;
only_1 = src_1_neighbors;

% 
only_centers = speye(size(only_1,1));

% 
src_1_neighbors = only_centers + only_1*0.8;
src_2_neighbors = only_centers + only_1*0.8 + only_2*0.60;
src_3_neighbors = only_centers + only_1*0.8 + only_2*0.60 + only_3*0.4;

neighbors_info.src_1_neighbors = src_1_neighbors;
neighbors_info.src_2_neighbors = src_2_neighbors;
neighbors_info.src_3_neighbors = src_3_neighbors;

%% source generation parameter setting.
use_guassian_basis = false;
pm_source = [];
pm_source.use_AR = false; 
pm_source.frequency = 100;
pm_source.recording_time = 0.01;
pm_source.addSourceSpaceNoise = false;
pm_source.noise_type = 'Guassian';
% pm_source.SNR_source = 10000;
pm_source.spike_number = 3;
pm_source.addSensorNoise = false;
pm_source.SNR_sensor = 10000;
pm_source.SpatialSmooth = use_guassian_basis;


%% Generate data part...产生数据的部分

EEG = [];
source = [];
normalize_L = true; % ==================== change here to normalize leadfield matrix ====================

% region index 中心域索引
left_index = (1:1:1026);
right_index = (1026+1:1:2052);
all_index_set=[left_index, right_index];

% set number of sample data 设置样本数量
total_data = 100;

% set region 设置激活区域
for regions = [2]%[2 1] % ==================== change here ====================

    if regions==1
        regions_index_set = sort(randperm(length(all_index_set),total_data)); 
    else % regions>=2
        region_left = floor(regions/2);
        region_right = regions-region_left;
        regions_index_set = [];
        for idx=1:regions
            regions_index_temp = randperm(length(left_index),total_data);
            regions_index_set = [regions_index_set; regions_index_temp];
        end
        regions_index_set(region_left+1:regions,:) = regions_index_set(region_left+1:regions,:) + length(left_index);
    end

    regions_index_set = regions_index_set';

    for neighbors = [1]%[3 2 1] % ==================== change here ====================
        for SNR_source = [10000]
            if SNR_source == 10000
                pm_source.addSourceSpaceNoise = false;
            else
                pm_source.addSourceSpaceNoise = true;
            end

            pm_source.SNR_source = SNR_source;

            for SNR_sensor = [10]%[40 30 20 10 0] % ==================== change here ====================
                if SNR_sensor == 10000
                    pm_source.addSensorNoise = false;
                else
                    pm_source.addSensorNoise = true; % true
                end

                pm_source.SNR_sensor = SNR_sensor;

                leadfield = load('new_leadfield.mat', 'leadfield');
                leadfield = leadfield.leadfield/1000;

                for i = 1:1:size(regions_index_set,1)
                    regions_index = regions_index_set(i,:);%  ==================== set region index ====================

                    if use_guassian_basis == 1
                        guassian_basis = get_guassian_basis(G_short_dist_mat);
                        leadfield_guassian = leadfield * guassian_basis;
                    else
                        leadfield_guassian = leadfield;
                    end

                    %
                    [M,N] = size(leadfield_guassian);% 128*2052
                    M_ones = ones(M,1);% 128*1
                    N_normalizer = sqrt(sum(leadfield_guassian.*leadfield_guassian));% 1*2052

                    if normalize_L == true
                        matrix_norm_den = M_ones*N_normalizer;% 128*2052
                        leadfield_guassian = leadfield_guassian./matrix_norm_den;
                    else
                        leadfield_guassian = leadfield_guassian;
                    end

                    L = leadfield_guassian;
                    save('Lead_Field.mat', 'L');% 128*2052

                    [X, S] = generate_data_region_all_neighbors_all_AR_all(leadfield_guassian, neighbors_info, regions_index, pm_source, regions, neighbors);

                    EEG = [EEG, X];
                    source = [source, S];
                end

                file_name = [data_save_path,'dataset_regions_',num2str(regions),'_neighbors_',num2str(neighbors),'_SNR_Source_',num2str(SNR_source), '_SNR_Sensor_', num2str(SNR_sensor), '.mat'];
                save(file_name , 'EEG', 'source');
                EEG = [];
                source = [];

            end
        end
    end
end
