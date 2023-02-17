%Low pass filter for the cluster voltages
wc=2*pi*200;             %cut off frequency of LPFs
sw_bw = 1;
wp=2*pi*10;

%Grid 
f_grid=50;
Vfn = 100;
Lg=1.0e-3; %1
%load
P=1.5e3;
%MMC
L = 5e-3;
E = 250;
N=3;
C_HB = 5e-3;  %3.3
VC0_HB = 1.2*E/N;
%C_HB_vector = C_HB*ones(1,N);
%VC0_HB_vector = VC0_HB*ones(1,N);
%VC0_HB_vector = VC0_HB*[0.9 1.1 1.15];

%Plantas de balance de capas

Kp=3*50;
Ki=1*19;

Kn1=-Vfn/(4*C_HB*VC0_HB); %0 SIGMA
Kn2=E/(2*C_HB*VC0_HB); %ALFA BETA SIGMA
Kn3=-Vfn/(C_HB*VC0_HB); %ALFA DELTA
Kn4=Vfn/(C_HB*VC0_HB); %BETA DELTA
Kn5=-Vfn/(C_HB*VC0_HB); %0 DELTA

%Modulation Controllers
Kb = 0.1;

fc = 8e3; %carrier frequency
PS = 0*[0:1/(N*fc):(N-1)/(N*fc)];

%KALMAN FILTER PARAMETERS
A=1;
B=1;
C=1;

R=0.001;    %VARIANZA
Q=0.5;    %VARIANZA
T=1/fc;

%%CONSENSUS PARAMETERS
Kf=-3;  %-3
ct=1;

%STEP TIME
ST=100;
m_max=1;
m_min=-1;