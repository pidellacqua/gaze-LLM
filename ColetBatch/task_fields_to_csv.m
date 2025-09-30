function export_task_fields_to_csv(Data, outdir)
% Esporta per ogni partecipante i campi task.gaze, task.pupil, task.blinks, task.annotation
% in file CSV: participant_XX_gaze.csv, ... nella cartella outdir (default: 'task_csv').

    if nargin < 2 || isempty(outdir)
        outdir = 'task_csv';
    end
    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    fields = {'gaze','pupil','blinks','annotation'};
    N = numel(Data);

    for i = 1:N
        for f = 1:numel(fields)
            fld = fields{f};
            for j = 1:4
                if isfield(Data(i).task(j), fld)
                    val = Data(i).task(j).(fld);
                    fname = fullfile(outdir, sprintf('participant_%02d_%02d_%s.csv', i, j, fld));
                    try
                        writetable(ensure_table(val), fname);  % usa writetable come richiesto
                        fprintf('✅ %s %2d -> %s\n', fld, i, fname);
                    catch ME
                        warning('⚠️  Impossibile salvare %s per partecipante %d, prova %d : %s', fld, i,j, ME.message);
                    end
                else
                    warning('⚠️  Campo "%s" assente per partecipante %d, prova %d .', fld, i, j);
                end
            end
        end
    end
end

function T = ensure_table(x)
% Converte x in table se non lo è già, così da usare sempre writetable.

    if istable(x)
        T = x;
        return;
    end
    if isstruct(x)
        T = struct2table(x);
        return;
    end
    if isnumeric(x) || islogical(x)
        T = array2table(x);
        return;
    end
    if iscell(x)
        % prova cell2table; se fallisce, serializza ogni cella in stringa
        try
            T = cell2table(x);
        catch
            S = cellfun(@toStringSafe, x, 'UniformOutput', false);
            T = table(S, 'VariableNames', {'value'});
        end
        return;
    end
    % fallback: una colonna con rappresentazione testuale
    T = table(string(toStringSafe(x)), 'VariableNames', {'value'});
end

function s = toStringSafe(v)
    try
        if isstring(v) || ischar(v), s = string(v); return; end
        if isnumeric(v) || islogical(v), s = string(v); return; end
        if istable(v), s = string(jsonencode(table2struct(v))); return; end
        if isstruct(v) || iscell(v), s = string(jsonencode(v)); return; end
        s = string(jsonencode(v));
    catch
        s = "<unserializable>";
    end
end

export_task_fields_to_csv(Data, 'colet_task_csv_2');