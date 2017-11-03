directoryContents = dir;
filenames = {directoryContents.name};
for i = 1:length(filenames)
    filename = filenames{i};
    if ~endsWith(filename, '.csv')
        continue
    end
    data = csvread(filename);
    modelname = filename(1:strfind(filename, '-')-1);
    figure;
    hold on;
    plot(data(:,1));
    plot(data(:, 2));
    xlabel('Epochs');
    ylabel('Accuracy');
    legend('Training Accuracy', 'Testing Accuracy');
    title([modelname, ' Performance Analysis']);
    savefig([modelname, '-analysis.fig']);
end