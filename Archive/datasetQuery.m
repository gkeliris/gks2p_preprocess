function ds = datasetQuery(varargin)

% set up name-value pairs for varargin.
for i = 1:2:length(varargin) % work for a list of name-value pairs
    if ischar(varargin{i}) % check if is character
        prm.(varargin{i}) = varargin{i+1}; % override or add parameters to structure.
    end
end
prm.make = 1;
% read the table from the CSV file
T = readtable('/mnt/NAS_UserStorage/georgioskeliris/MECP2TUN/MECP2_datasets.csv','TextType','string');

if ~isfield(prm, 'cohort')
    prm.cohort = unique(T.cohort);
end
if ~isfield(prm, 'week')
    prm.week = unique(T.week);
end
if ~isfield(prm, 'mouseID')
    prm.mouseID = unique(T.mouseID);
end
if ~isfield(prm, 'session')
    prm.session = unique(T.session);
end
if ~isfield(prm, 'expID')
    prm.expID = unique(T.expID);
else
    ex=prm.expID;
    prm.expID = {ex, [ex '1'], [ex '2'], [ex '3'], [ex '4'], [ex '5'], [ex '6']}';
end


ds = T(contains(T.cohort,prm.cohort) & contains(T.week,prm.week) & ...
    contains(T.mouseID,prm.mouseID) & contains(T.session,prm.session) & ...
    contains(T.expID, prm.expID),:);
