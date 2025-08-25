% Number of data generated
saveFolder = 'dataset'; 
imageDataFolder = fullfile(saveFolder, 'Images');
labelDataFolder = fullfile(saveFolder, 'Labels');
if ~exist(imageDataFolder, 'dir')
    mkdir(imageDataFolder);
end
if ~exist(labelDataFolder, 'dir')
    mkdir(labelDataFolder);
end
numTrainingData=20000;
imageSize=[256 256];

% Define data directory
classNames = ["Noise" "LTE" "NR" "Radar"];
pixelLabelID = [0 1 2 3];

% Consider an airport surveillance radar located at the scenario origin. 
% The radar operates at 2.8 GHz and uses a reflector antenna with a gain of 32.8 dB. 
% The transmit power is set at 25 kW. 

% setup waveform
fc=2.8e9;                   % center frequency [Hz]
fs=61.44e6;                 % sampling frequency [Hz]
prf=fs/ceil(fs/1050);       % pulse repetition rate [Hz]
pulseWidth=1e-06;           % pulsewdith [s]
wav = phased.RectangularWaveform('SampleRate',fs,'PulseWidth',pulseWidth,'PRF',prf,'NumPulses',3);

% setup antenna
rad=2.5;            % radius [m]
flen=2.5;           % focal length [m]
antele=design(reflectorParabolic('Exciter',horn),fc);
antele.Exciter.Tilt=90;
antele.Exciter.TiltAxis=[0 1 0];
antele.Tilt=90;
antele.TiltAxis=[0 1 0];
antele.Radius=rad;
antele.FocalLength=flen;
ant=phased.ConformalArray('Element',antele);

% setup transmitter and receiver
power_ASR=25000;    % transmit power [W]
gain_ASR=32.8;      % transmit gain [dB]
radartx = phased.Transmitter('PeakPower',power_ASR,'Gain',gain_ASR);

% For the wireless signals, the gain and the power of the transmitter may change from frame to frame. 
% The receivers are assumed to be randomly placed in a 2 km x 2km region and equipped with isotropic antennas.
rxpos_horiz_minmax=[-1000 1000];
rxpos_vert_minmax=[0 2000];

% The region also contains many scatterers which makes the propagation channels more challenging to operate in. 
% In this example, we assume the channel contains 30 scatterers, randomly distributed in the region. 
% You can use phased.ScatteringMIMOChannel to simulate the scatterers.
% define radar position
radartxpos=[0 0 15]';
radartxaxes=rotz(0);
radarchan=phased.ScatteringMIMOChannel('TransmitArray',ant,... 
    'ReceiveArray',phased.ConformalArray('Element',phased.IsotropicAntennaElement),...
    'CarrierFrequency',fc,...
    'SpecifyAtmosphere',true,...
    'SampleRate',fs,...
    'SimulateDirectPath',false,...
    'MaximumDelaySource','Property',...
    'MaximumDelay',2e-5,...
    'TransmitArrayMotionSource','Property',...
    'TransmitArrayPosition',radartxpos,...
    'TransmitArrayOrientationAxes',radartxaxes,...
    'ReceiveArrayMotionSource','Input port',...
    'ScattererSpecificationSource','Input port');

% define wireless transmitter position
commtxpos=[200 0 450]';
commtxaxes=rotz(0);
commchan=phased.ScatteringMIMOChannel('TransmitArray',phased.ConformalArray('Taper',10),... 
    'ReceiveArray',phased.ConformalArray('Element',phased.IsotropicAntennaElement),...
    'CarrierFrequency',fc,...
    'SpecifyAtmosphere',true,...
    'SampleRate',fs,...
    'SimulateDirectPath',false,...
    'MaximumDelaySource','Property',...
    'MaximumDelay',2e-5,...
    'TransmitArrayMotionSource','Property',...
    'TransmitArrayPosition',commtxpos,...
    'TransmitArrayOrientationAxes',commtxaxes,...
    'ReceiveArrayMotionSource','Input port',...
    'ScattererSpecificationSource','Input port');

if ~isempty(gcp('nocreate'))
    delete(gcp('nocreate')); % Shut down any existing pool
end
parpool('local', 4); % Start a new pool with 4 workers
helperGenerateRadarCommData(fs, wav, radartx, radarchan, commchan, rxpos_horiz_minmax, rxpos_vert_minmax, numTrainingData, imageDataFolder, labelDataFolder, imageSize);

% dataURL = 'https://ssd.mathworks.com/supportfiles/phased/data/RadarCommSpectrumSensingData.zip';
% zipFile = fullfile(saveFolder,'RadarCommSpectrumSensingData.zip');
