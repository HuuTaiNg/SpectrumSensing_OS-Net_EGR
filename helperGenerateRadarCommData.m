function helperGenerateRadarCommData(fs, wav, radartx, radarchan, commchan, rxpos_horiz_minmax, rxpos_vert_minmax, numData, imageDir, labelDir, imageSize)
%HELPERGENERATERADARCOMMDATA Generates and saves radar and communication signal spectrograms.
%
% This function was modified to include a regenerative loop to ensure that
% for every iteration, a valid combination of 5G and LTE signals is found
% that fits within the specified sampling rate. It also saves the output
% spectrograms (images) and their corresponding labels into separate
% directories as specified by imageDir and labelDir, with a custom filename structure.

% Transmit power and gain parameters for communication signals
power_5G_LTE=350; % transmit power [W]
gain_5G_LTE=10;   % transmit gain [dB]

% Channel and signal parameters
maxNumScat=30;
SNRdBVec = {-10,0,10,20,30};
doppler_shifts = [0, 5, 70, 500]; % Doppler shift values (in Hz)
c = 299792458;                    % Speed of light (m/s)
fc = 2.8e9;                       % Radar carrier frequency (Hz)

parfor highestIndex=1:numData
    %% Create airport surveillance radar (ASR) waveform
    pulses = wav();
    % Introduce a random start time for the pulse
    numZeros=randi([1 round(wav.PulseWidth*wav.SampleRate*3/4)]);
    pulses=[zeros(1,numZeros) pulses.'];

    %% Create 5G and LTE waveform with valid bandwidth
    % This section uses a 'while' loop to ensure the combined bandwidth of
    % the randomly generated 5G and LTE signals does not exceed the
    % system's sampling rate.

    % General and LTE/5G Parameters
    numSubFrames = 40;
    maxTimeShift = numSubFrames;
    RCVec = {'R.6','R.8','R.9'};
    TrBlkOffVec = {1,2,3,4,5,6,7,8};
    SCSVec = [15 30];
    BandwidthVec = [10:5:30 40 50];
    SSBPeriodVec = 20;

    % Initialize with a value that fails the check to ensure the loop runs at least once
    maxFreqSpace = -1;

    while maxFreqSpace <= 0
        % Generate LTE signal
        RC = RCVec{randi([1 length(RCVec)])};
        timeShift_LTE = rand() * maxTimeShift;
        TrBlkOff = TrBlkOffVec{randi([1 length(TrBlkOffVec)])};
        [txWaveLTE, waveinfoLTE] = helperGenerateRadarCommLTESignal(RC, timeShift_LTE, TrBlkOff, numSubFrames, fs);
        x1_LTE = txWaveLTE;

        % Generate 5G Signal
        % scs = SCSVec(randi([1 length(SCSVec)]));
        % nrChBW = BandwidthVec(randi([1 length(BandwidthVec)]));
        validCombinations = [
            % SCS=15kHz Combinations
            10, 15; 15, 15; 20, 15; 25, 15; 30, 15; 40, 15; 50, 15;
            % SCS=30kHz Combinations
            10, 30; 15, 30; 20, 30; 25, 30; 30, 30; 40, 30; 50, 30;
            % (Add any other valid combinations you need from the table)
        ];
        
        % Randomly select a valid row from the list
        rowIndex = randi(size(validCombinations, 1));
        nrChBW = validCombinations(rowIndex, 1);
        scs = validCombinations(rowIndex, 2);

        timeShift_5G = rand() * maxTimeShift;
        SSBPeriod = SSBPeriodVec(randi([1 length(SSBPeriodVec)]));
        [txWave5G, waveinfo5G] = helperGenerateRadarCommNRSignal(scs, nrChBW, SSBPeriod, timeShift_5G, numSubFrames, fs);
        x1_5G = txWave5G;

        % Check if the combined bandwidth fits within the sampling rate
        if ~isempty(x1_5G)
            maxFreqSpace = fs - (waveinfo5G.Bandwidth + waveinfoLTE.Bandwidth);
        else
            % If 5G signal generation failed, force a re-run
            maxFreqSpace = -1;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%

    %% Channel Simulation
    
    % Randomly define scatterer positions and coefficients for the channel
    numScat=randi([1 maxNumScat]);
    scatPos_x=rand([numScat 1])*2000-1000;
    scatPos_y=rand([numScat 1])*2000-1000;
    scatPos_z=rand([numScat 1])*1000;
    scatPos=[scatPos_x'; scatPos_y'; scatPos_z';];
    scatCoef_real=rand([numScat 1])*4-2;
    scatCoef_imag=rand([numScat 1])*4-2;
    scatCoef=scatCoef_real+1i*scatCoef_imag;

    % Randomly define receiver position
    rec_xy=randn([2 1])*(rxpos_horiz_minmax(2)-rxpos_horiz_minmax(1))+rxpos_horiz_minmax(1);
    rec_z=rand([1 1])*(rxpos_vert_minmax(2)-rxpos_vert_minmax(1))+rxpos_vert_minmax(1);

    % Select a random Doppler shift and calculate receiver velocity
    doppler_shift = doppler_shifts(randi(length(doppler_shifts)));
    receiver_velocity_magnitude = doppler_shift * c / (2 * fc);
    angle = 2 * pi * rand();
    receiver_velocity = receiver_velocity_magnitude * [cos(angle); sin(angle); 0];


    % Transmit ASR signal through the radar channel
    y=radartx(pulses');
    release(radarchan);
    y=radarchan(y,[rec_xy;rec_z],receiver_velocity,eye(3),scatPos,zeros(size(scatPos)),scatCoef');

    % Transmit 5G and LTE signals through the communication channel
    p_5G_LTE=power_5G_LTE(randi(length(power_5G_LTE)));
    g_5G_LTE=gain_5G_LTE(randi(length(gain_5G_LTE)));
    commtx=phased.Transmitter('PeakPower',p_5G_LTE,'Gain',g_5G_LTE);
    
    % Transmit 5G signal
    if ~isempty(x1_5G)
        y1_5G = commtx(x1_5G);
        release(commchan);
        y1_5G = commchan(y1_5G,[rec_xy;rec_z],receiver_velocity,eye(3),scatPos,zeros(size(scatPos)),scatCoef');
    else
        waveinfo5G.Bandwidth=[];
        y1_5G = [eps zeros(1,length(y(:,1)))]';
    end
    
    % Transmit LTE signal
    y1_LTE = commtx(x1_LTE);
    release(commchan);
    y1_LTE = commchan(y1_LTE,[rec_xy;rec_z],receiver_velocity,eye(3),scatPos,zeros(size(scatPos)),scatCoef');


    %% Combining signals

    % Ensure all signals have the same length
    length_sig = min([length(y) length(y1_5G) length(y1_LTE)]);
    y=y(1:length_sig);
    
    [~,index_5g]=max(y1_5G(1:(end-length_sig)));
    if ~isempty(index_5g)
        y1_5G=y1_5G(index_5g:(index_5g-1+length_sig));
    end
    
    [~,index_lte]=max(y1_LTE(1:(end-length_sig)));
    if ~isempty(index_lte)
        y1_LTE=y1_LTE(index_lte:(index_lte-1+length_sig));
    end

    % Combine 5G and LTE signals with random frequency offset
    if ~isempty(x1_5G)
        sr = fs;
        comb = comm.MultibandCombiner("InputSampleRate",sr, ...
          "OutputSampleRateSource","Property",...
          "OutputSampleRate",sr);
        
        maxFreqSpace = sr - (waveinfo5G.Bandwidth + waveinfoLTE.Bandwidth);
        freqSpace = round(rand()*maxFreqSpace/1e6)*1e6;
        freqPerPixel = sr / imageSize(2);
        maxStartFreq = sr - (waveinfo5G.Bandwidth + waveinfoLTE.Bandwidth + freqSpace) - freqPerPixel;
        
        LTEFirst = randi([0 1]);
        if LTEFirst
          combIn = [y1_LTE, y1_5G];
          startFreq = round(rand()*maxStartFreq/1e6)*1e6 - sr/2 + waveinfoLTE.Bandwidth/2;
        else
          combIn = [y1_5G, y1_LTE];
          startFreq = round(rand()*maxStartFreq/1e6)*1e6 - sr/2 + waveinfo5G.Bandwidth/2;
        end
        release(comb)
        comb.FrequencyOffsets = [startFreq startFreq+waveinfoLTE.Bandwidth/2 + freqSpace + waveinfo5G.Bandwidth/2];
        txWave_5G_LTE = comb(combIn);
    end

    % Combine ASR signal with communication signals
    y_ASR_5G=y+y1_5G;
    y_ASR_LTE=y+y1_LTE;
    if ~isempty(x1_5G)
        y_ASR_5G_LTE=y+txWave_5G_LTE;
    end

    % Add Additive White Gaussian Noise (AWGN)
    snr_idx = randi(length(SNRdBVec));
    SNRdB = SNRdBVec{snr_idx};
    y_ASR_5G_gauss = awgn(y_ASR_5G,SNRdB,'measured');
    y_ASR_LTE_gauss = awgn(y_ASR_LTE,SNRdB,'measured');
    if ~isempty(x1_5G)
        y_ASR_5G_LTE_gauss = awgn(y_ASR_5G_LTE,SNRdB,'measured');
    end

    %% Generate and Save Spectrograms and Labels

    % Spectrogram parameters
    overlap = 10;
    Nfft = 4096;
    sr = fs;

    % --- Save ASR and 5G ---
    saveSpectrogram(y_ASR_5G, ...
        y_ASR_5G_gauss, ...
        waveinfo5G.Bandwidth, imageSize, highestIndex, y, length_sig, ...
        overlap, Nfft, sr, imageDir, labelDir, false, snr_idx, doppler_shift);

    % --- Save ASR and LTE ---
    saveSpectrogram(y_ASR_LTE, ...
        y_ASR_LTE_gauss, ...
        waveinfoLTE.Bandwidth, imageSize, highestIndex, y, length_sig, ...
        overlap, Nfft, sr, imageDir, labelDir, true, snr_idx, doppler_shift);

    % --- Save ASR and 5G and LTE ---
    if ~isempty(x1_5G)
        saveSpectrogram1(y_ASR_5G_LTE, ...
            y_ASR_5G_LTE_gauss, ...
            waveinfo5G.Bandwidth, waveinfoLTE.Bandwidth, imageSize, highestIndex, y, length_sig, ...
            overlap, Nfft, sr, imageDir, labelDir, LTEFirst, comb.FrequencyOffsets, snr_idx, doppler_shift);
    end
end
end


%% Helper Functions to Save Spectrograms and Labels

function saveSpectrogram(signal, signalNoisy, waveBW, imageSize, highestIndex, ...
    y, length_sig, overlap, Nfft, sr, imageDir, labelDir, LTElabel, snr_idx, doppler_shift)

    % Determine signal type
    if LTElabel
        signal_type = 'LTE';
    else
        signal_type = '5G';
    end

    % --- Filename Generation ---
    % Clean files
    img_clean_fname = sprintf('data_%05d_radar_%s_%d_clean.png', highestIndex, signal_type, doppler_shift);
    lbl_clean_fname = sprintf('label_%05d_radar_%s_%d_clean.png', highestIndex, signal_type, doppler_shift);
    
    % Noisy files
    img_noisy_fname = sprintf('data_%05d_radar_%s_%d_snr_%d.png', highestIndex, signal_type, doppler_shift, snr_idx);
    lbl_noisy_fname = sprintf('label_%05d_radar_%s_%d_snr_%d.png', highestIndex, signal_type, doppler_shift, snr_idx);

    % --- Full File Paths ---
    imageDataPath = fullfile(imageDir, img_clean_fname);
    labelPath = fullfile(labelDir, lbl_clean_fname);
    imageDataPathNoisy = fullfile(imageDir, img_noisy_fname);
    labelPathNoisy = fullfile(labelDir, lbl_noisy_fname);
    
    % --- Spectrogram (Clean) ---
    [~,~,~,P] = spectrogram(signal,hann(256),overlap, Nfft,sr,'centered','psd','xaxis');
    P = 10*log10(abs(P)'+eps);
    im = ind2rgb(im2uint8(rescale(P)),parula(256));
    rxSpectrogram = im2uint8(flipud(imresize(im,imageSize,"nearest")));
    imwrite(rxSpectrogram, imageDataPath);

    % --- Spectrogram (Noisy) ---
    [~,~,~,P] = spectrogram(signalNoisy,hann(256),overlap, Nfft,sr,'centered','psd','xaxis');
    P = 10*log10(abs(P)'+eps);
    im = ind2rgb(im2uint8(rescale(P)),parula(256));
    rxSpectrogram = im2uint8(flipud(imresize(im,imageSize,"nearest")));
    imwrite(rxSpectrogram, imageDataPathNoisy);

    % --- Label Generation (for both clean and noisy) ---
    labelData=zeros(imageSize, 'uint8');
    freqPerPixel=sr/imageSize(2);
    BW_pixels=waveBW/freqPerPixel;
    idx_BW_1=round(imageSize(2)/2-BW_pixels/2);
    idx_BW_2=round(imageSize(2)/2+BW_pixels/2);
    if LTElabel
     labelData(:,idx_BW_1:idx_BW_2)=80;  % LTE = 80
    else
     labelData(:,idx_BW_1:idx_BW_2)=160; % 5G = 160
    end
    indices_pulses=find(y);
    indices_spectrogram=unique(ceil(indices_pulses*imageSize(1)/length_sig));
    labelData(indices_spectrogram,:)=255; % Radar = 255
    labelData=flipud(labelData);
    
    % Save labels
    imwrite(labelData, labelPath);
    imwrite(labelData, labelPathNoisy);
end


function saveSpectrogram1(signal, signalNoisy, waveBW_5G, waveBW_LTE, imageSize, ...
    highestIndex, y, length_sig, overlap, Nfft, sr, imageDir, labelDir, ...
    LTEFirst, freqOffset, snr_idx, doppler_shift)

    % Define signal type
    signal_type = '5G_LTE';

    % --- Filename Generation ---
    % Clean files
    img_clean_fname = sprintf('data_%05d_radar_%s_%d_clean.png', highestIndex, signal_type, doppler_shift);
    lbl_clean_fname = sprintf('label_%05d_radar_%s_%d_clean.png', highestIndex, signal_type, doppler_shift);
    
    % Noisy files
    img_noisy_fname = sprintf('data_%05d_radar_%s_%d_snr_%d.png', highestIndex, signal_type, doppler_shift, snr_idx);
    lbl_noisy_fname = sprintf('label_%05d_radar_%s_%d_snr_%d.png', highestIndex, signal_type, doppler_shift, snr_idx);

    % --- Full File Paths ---
    imageDataPath = fullfile(imageDir, img_clean_fname);
    labelPath = fullfile(labelDir, lbl_clean_fname);
    imageDataPathNoisy = fullfile(imageDir, img_noisy_fname);
    labelPathNoisy = fullfile(labelDir, lbl_noisy_fname);

    % --- Spectrogram (Clean) ---
    [~,~,~,P] = spectrogram(signal,hann(256),overlap, Nfft,sr,'centered','psd','xaxis');
    P = 10*log10(abs(P)'+eps);
    im = ind2rgb(im2uint8(rescale(P)),parula(256));
    rxSpectrogram = im2uint8(flipud(imresize(im,imageSize,"nearest")));
    imwrite(rxSpectrogram, imageDataPath);

    % --- Spectrogram (Noisy) ---
    [~,~,~,P] = spectrogram(signalNoisy,hann(256),overlap, Nfft,sr,'centered','psd','xaxis');
    P = 10*log10(abs(P)'+eps);
    im = ind2rgb(im2uint8(rescale(P)),parula(256));
    rxSpectrogram = im2uint8(flipud(imresize(im,imageSize,"nearest")));
    imwrite(rxSpectrogram, imageDataPathNoisy);

    % --- Label Generation (for both clean and noisy) ---
    labelData=zeros(imageSize, 'uint8');
    freqPerPixel=sr/imageSize(2);
    freqOffset_pixels=freqOffset/freqPerPixel;
    BW_pixels_5G=waveBW_5G/freqPerPixel;
    BW_pixels_LTE=waveBW_LTE/freqPerPixel;
    if LTEFirst
        idx_BW_1=round(imageSize(2)/2+freqOffset_pixels(1)-BW_pixels_LTE/2);
        idx_BW_2=round(imageSize(2)/2+freqOffset_pixels(1)+BW_pixels_LTE/2);
        idx_BW_3=round(imageSize(2)/2+freqOffset_pixels(2)-BW_pixels_5G/2);
        idx_BW_4=round(imageSize(2)/2+freqOffset_pixels(2)+BW_pixels_5G/2);
        if idx_BW_1<1, idx_BW_1=1; end
        if idx_BW_4>imageSize(2), idx_BW_4=imageSize(2); end
        labelData(:,idx_BW_1:idx_BW_2)=80;      % LTE = 80
        labelData(:,idx_BW_3:idx_BW_4)=160;     % 5G = 160
    else
        idx_BW_1=round(imageSize(2)/2+freqOffset_pixels(1)-BW_pixels_5G/2);
        idx_BW_2=round(imageSize(2)/2+freqOffset_pixels(1)+BW_pixels_5G/2);
        idx_BW_3=round(imageSize(2)/2+freqOffset_pixels(2)-BW_pixels_LTE/2);
        idx_BW_4=round(imageSize(2)/2+freqOffset_pixels(2)+BW_pixels_LTE/2);
        if idx_BW_1<1, idx_BW_1=1; end
        if idx_BW_4>imageSize(2), idx_BW_4=imageSize(2); end
        labelData(:,idx_BW_1:idx_BW_2)=160;     % 5G = 160
        labelData(:,idx_BW_3:idx_BW_4)=80;      % LTE = 80
    end
    indices_pulses=find(y);
    indices_spectrogram=unique(ceil(indices_pulses*imageSize(1)/length_sig));
    labelData(indices_spectrogram,:)=255;      % Radar = 255
    labelData=flipud(labelData);
    
    % Save labels
    imwrite(labelData, labelPath);
    imwrite(labelData, labelPathNoisy);
end