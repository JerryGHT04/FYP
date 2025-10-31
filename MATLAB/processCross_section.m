% Source folder
srcDir = fullfile('..', 'Cross-sections');
files  = dir(srcDir);
files  = files(~[files.isdir]);           % keep only files
filename = {files.name};

% Output folder: ../modified_cross_sections
outDir = fullfile(srcDir, '..', 'modified_cross_sections');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Preallocate container (optional)
data = cell(numel(filename), 1);

for i = 1:numel(filename)
    fpath = fullfile(files(i).folder, filename{i});

    % 1) Load cross-section data (col1: sqrt(eV), col2: 1e-20 m^2)
    M = readmatrix(fpath);

    % Guard: keep only first two numeric columns
    M = M(:, 1:2);

    % 2) Convert to eV and m^2
    M(:,1) = M(:,1).^2;         % sqrt(eV) -> eV
    M(:,2) = M(:,2) * 1e-20;    % (1e-20 m^2) -> m^2

    data{i} = M;

    % 3) Ensure output folder exists (already done above)

    % 4) Write with two spaces between columns, same filename
    outPath = fullfile(outDir, filename{i});
    fid = fopen(outPath, 'w');
    assert(fid ~= -1, 'Cannot open file for writing: %s', outPath);

    % Write each row as: <col1>␠␠<col2>
    for r = 1:size(M,1)
        fprintf(fid, '%.10g  %.10g\n', M(r,1), M(r,2));
    end
    fclose(fid);
end