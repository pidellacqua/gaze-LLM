function export_subjects_info(Data, outFile)
% Esporta i campi subject_info di tutti i partecipanti in un unico CSV.
%
% USO:
%   export_subjects_info(Data, 'subjects_data.csv')

    if nargin < 2 || isempty(outFile)
        outFile = 'subjects_data.csv';
    end

    N = numel(Data);
    allSubjects = cell(N,1);

    % raccogli subject_info di ciascun partecipante
    for i = 1:N
        allSubjects{i} = Data(i).subject_info;
    end

    % Se subject_info è uno struct -> concatenalo in tabella
    if isstruct(allSubjects{1})
        % converte struct array in tabella
        T = struct2table([allSubjects{:}]);
    elseif istable(allSubjects{1})
        % concatena tabelle già pronte
        T = vertcat(allSubjects{:});
    else
        error('subject_info non è né struct né table, impossibile esportare direttamente');
    end

    % aggiungi eventualmente colonna con indice partecipante
    T.participant_id = (1:N)';

    % salva in CSV
    writetable(T, outFile);

    fprintf('✅ Esportati %d subject_info in %s\n', N, outFile);
end

export_subjects_info(Data, 'subjects_data.csv');