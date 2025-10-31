files = dir('../Cross-sections/');
filename = {files.name};
filename = filename(3:end);  % skip '.' and '..'

data = cell(length(filename), 1);  % preallocate

for i = 1:length(filename)
    fpath = fullfile(files(1).folder, filename{i});
    data{i} = readmatrix(fpath);
end

% some constants
u = 1.66053906660e-27;
mO = 15.999*u;
mN = 14.007*u;
mO2 = mO*2;
mN2 = mN*2;
e = 1.60217663e-19;
V = 7790;

% energy of fast species etc. in eV
EN2 = 1/2*mN2*V^2/e;
EO2 = 1/2*mO2*V^2/e;
EO = 1/2*mO*V^2/e;

%plot resonant charge exchange cross-sections
figure(1)
clf;
plot()