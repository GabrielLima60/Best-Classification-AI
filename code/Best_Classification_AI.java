import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.List;
import javax.swing.text.NumberFormatter;
import java.text.NumberFormat;

public class Best_Classification_AI extends JFrame implements ActionListener {
    private JFileChooser fileChooser;
    private JPanel currentPage;
    private JPanel csvSelectionPage;
    private JPanel analysisConfigPage;
    private JPanel loadingPage;
    private JButton selectCSVButton;
    private JButton analyzeButton;
    private List<JCheckBox> modelCheckBoxes;
    private List<JCheckBox> techniqueCheckBoxes;
    private List<JCheckBox> parameterCheckBoxes;
    private JLabel techniquesLabel;
    private JLabel modelsLabel;
    private JLabel topLoadingLabel;
    private JLabel bottomLoadingLabel;
    private StringBuilder pythonOutput = new StringBuilder();

    private List<String> techniques = Arrays.asList("PCA", "IncPCA", "ICA", "LDA", "SMOTE");
    private List<String> models = Arrays.asList("SVM", "MLP", "DecisionTree", "RandomForest", "KNN", "LogReg", "GradientBoost", "XGBoost");
    private String selectedOptimization;
    private String selectedCrossValidation;
    private JFormattedTextField iterationsField;
    private int numberOfIterations;
    private List<String> parameters = Arrays.asList("F1-Score", "Processing Time", "ROC AUC", "Memory Usage", "Precision", "Accuracy","Recall");

    public Best_Classification_AI() {
        super("Best Classification AI");

        // Set look and feel to system default
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Initialize file chooser
        fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            public boolean accept(File f) {
                return f.getName().toLowerCase().endsWith(".csv") || f.isDirectory();
            }

            public String getDescription() {
                return "CSV files (*.csv)";
            }
        });

        // Create CSV selection page
        createCSVSelectionPage();

        // Create analysis configuration page
        createAnalysisConfigPage();

        //Create loading page
        createLoadingPage();

        // Set initial page to CSV selection page
        currentPage = csvSelectionPage;

        // Set up main frame
        ImageIcon icon = new ImageIcon("resources\\icon.png");
        setIconImage(icon.getImage());
        setContentPane(currentPage);
        setPreferredSize(new Dimension(1280, 720));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void createCSVSelectionPage() {
        csvSelectionPage = new JPanel();
        csvSelectionPage.setLayout(new BorderLayout());
        ImageIcon backgroundImage = new ImageIcon("resources/background.gif");
        JLabel backgroundLabel = new JLabel(backgroundImage);
        backgroundLabel.setLayout(new BorderLayout());

        selectCSVButton = new JButton("Choose CSV File");
        selectCSVButton.addActionListener(this);
        selectCSVButton.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 40));
        selectCSVButton.setPreferredSize(new Dimension(400, 100));
        JPanel buttonPanel = new JPanel();
        buttonPanel.setOpaque(false);
        buttonPanel.add(selectCSVButton);
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(300, 0, 0, 0));

        backgroundLabel.add(buttonPanel, BorderLayout.CENTER);
        csvSelectionPage.add(backgroundLabel, BorderLayout.CENTER);
    }

    private void createAnalysisConfigPage() {
        analysisConfigPage = new JPanel(new BorderLayout());
        analysisConfigPage.setBackground(Color.WHITE);

        // Adding background image
        ImageIcon backgroundImage = new ImageIcon("resources/background.gif");
        JLabel backgroundLabel = new JLabel(backgroundImage);
        backgroundLabel.setLayout(new BorderLayout());

        JPanel configPanel = new JPanel(new GridLayout(0, 1));
        configPanel.setOpaque(false);
        configPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        // Models selection panel
        JPanel modelsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        modelsLabel = new JLabel("Select Models:");
        modelsLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        modelsPanel.add(modelsLabel);
        modelCheckBoxes = new ArrayList<>();
        for (String model : models) {
            JCheckBox checkBox = new JCheckBox(model);
            checkBox.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
            modelCheckBoxes.add(checkBox);
            modelsPanel.add(checkBox);
        }
        configPanel.add(modelsPanel);

        // Techniques selection panel
        JPanel techniquesPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        techniquesLabel = new JLabel("Select Techniques:");
        techniquesLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        techniquesPanel.add(techniquesLabel);
        techniqueCheckBoxes = new ArrayList<>();
        for (String technique : techniques) {
            JCheckBox checkBox = new JCheckBox(technique);
            checkBox.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
            techniqueCheckBoxes.add(checkBox);
            techniquesPanel.add(checkBox);
        }
        configPanel.add(techniquesPanel);

        // Optimization selection panel
        JPanel optimizationPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel optimizationLabel = new JLabel("Optimization:");
        optimizationLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        optimizationPanel.add(optimizationLabel);
        JRadioButton gridSearchRadioButton = new JRadioButton("Grid Search");
        gridSearchRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        JRadioButton randomSearchRadioButton = new JRadioButton("Random Search");
        randomSearchRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        //JRadioButton geneticAlgorithmRadioButton = new JRadioButton("Genetic Algorithm");
        //geneticAlgorithmRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        JRadioButton NoSearchRadioButton = new JRadioButton("None");
        NoSearchRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));

        ButtonGroup optimizationGroup = new ButtonGroup();
        optimizationGroup.add(gridSearchRadioButton);
        optimizationGroup.add(randomSearchRadioButton);
        //optimizationGroup.add(geneticAlgorithmRadioButton);
        optimizationGroup.add(NoSearchRadioButton);


        optimizationPanel.add(gridSearchRadioButton);
        optimizationPanel.add(randomSearchRadioButton);
        //optimizationPanel.add(geneticAlgorithmRadioButton);
        optimizationPanel.add(NoSearchRadioButton);

        gridSearchRadioButton.addActionListener(e -> selectedOptimization = "Grid Search");
        randomSearchRadioButton.addActionListener(e -> selectedOptimization = "Random Search");
        //geneticAlgorithmRadioButton.addActionListener(e -> selectedOptimization = "Genetic Algorithm");
        NoSearchRadioButton.addActionListener(e -> selectedOptimization = "None");

        gridSearchRadioButton.setSelected(true);
        selectedOptimization = "Grid Search";
        
        configPanel.add(optimizationPanel);

        // Cross Validation selection panel
        JPanel crossValidationPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel crossValidationLabel = new JLabel("Cross Validation:");
        crossValidationLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        crossValidationPanel.add(crossValidationLabel);
        JRadioButton kFoldRadioButton = new JRadioButton("K-Fold");
        kFoldRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        JRadioButton holdOutRadioButton = new JRadioButton("Hold-Out");
        holdOutRadioButton.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        ButtonGroup crossValidationGroup = new ButtonGroup();
        crossValidationGroup.add(kFoldRadioButton);
        crossValidationGroup.add(holdOutRadioButton);
        crossValidationPanel.add(kFoldRadioButton);
        crossValidationPanel.add(holdOutRadioButton);

        kFoldRadioButton.addActionListener(e -> selectedCrossValidation = "K-Fold");
        holdOutRadioButton.addActionListener(e -> selectedCrossValidation = "Hold-Out");

        kFoldRadioButton.setSelected(true);
        selectedCrossValidation = "K-Fold";

        configPanel.add(crossValidationPanel);

        // Number of iterections selection panel
        JPanel iterationsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel iterationsLabel = new JLabel("Number of Iterations:");
        iterationsLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        NumberFormat integerFormat = NumberFormat.getIntegerInstance();
        integerFormat.setGroupingUsed(false);
        NumberFormatter numberFormatter = new NumberFormatter(integerFormat);
        numberFormatter.setValueClass(Integer.class);
        numberFormatter.setMinimum(0);
        numberFormatter.setMaximum(999);
        numberFormatter.setAllowsInvalid(false);
        iterationsField = new JFormattedTextField(numberFormatter);
        iterationsField.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
        iterationsField.setColumns(10);
        iterationsField.setValue(10); // Default value
        iterationsPanel.add(iterationsLabel);
        iterationsPanel.add(iterationsField);
        configPanel.add(iterationsPanel);



        // Parameters Analysed selection panel
        JPanel parametersPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel parametersLabel = new JLabel("Parameters Analysed:");
        parametersLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 28));
        parametersPanel.add(parametersLabel);
        parameterCheckBoxes = new ArrayList<>();
        for (String parameter : parameters) {
            JCheckBox checkBox = new JCheckBox(parameter);
            checkBox.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 22));
            parameterCheckBoxes.add(checkBox);
            parametersPanel.add(checkBox);
        }
        configPanel.add(parametersPanel);

        // Analyze button panel
        analyzeButton = new JButton("Start Analysis");
        analyzeButton.addActionListener(this);
        analyzeButton.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 40));
        analyzeButton.setPreferredSize(new Dimension(400, 100));
        JPanel analyzePanel = new JPanel();
        analyzePanel.setOpaque(false);
        analyzePanel.add(analyzeButton);
        configPanel.add(analyzePanel);

        backgroundLabel.add(configPanel, BorderLayout.CENTER);
        analysisConfigPage.add(backgroundLabel, BorderLayout.CENTER);
    }

    private void createLoadingPage() {
        loadingPage = new JPanel(new BorderLayout());
        loadingPage.setBackground(Color.WHITE);

        // Adding background image
        ImageIcon backgroundImage = new ImageIcon("resources/background.gif");
        JLabel backgroundLabel = new JLabel(backgroundImage);
        backgroundLabel.setLayout(new BorderLayout());

        JPanel loadingPanel = new JPanel(new GridLayout(0, 1));
        loadingPanel.setOpaque(false);
        loadingPanel.setBorder(BorderFactory.createEmptyBorder(120, 80, 120, 80));

        JPanel textPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        topLoadingLabel = new JLabel("Loading...");
        topLoadingLabel.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 100));
        textPanel.add(topLoadingLabel);

        loadingPanel.add(textPanel);

        JPanel bottomTextPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        bottomLoadingLabel = new JLabel("<html>This analysis program may take hours to finish<br/>Leave it running in the background<br/><br/>Results will be available at Best-Classification-AI/results table<br/>The graphs will be available at Best-Classification-AI/results image</html>");
        bottomLoadingLabel.setFont(new Font("Lucida Sans Unicode", Font.PLAIN, 24));
        bottomTextPanel.add(bottomLoadingLabel);

        loadingPanel.add(bottomTextPanel);

        backgroundLabel.add(loadingPanel, BorderLayout.CENTER);
        loadingPage.add(backgroundLabel, BorderLayout.CENTER);
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == selectCSVButton) {
            openCSVSelection();
        } else if (e.getSource() == analyzeButton) {

            currentPage = loadingPage;
            setContentPane(currentPage);
            revalidate();
            repaint();

            // Start analysis in a background thread
            SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
                @Override
                protected Void doInBackground() {
                    performAnalysis();
                    return null;
                }
            };
            worker.execute();
        }
    }

    private void openCSVSelection() {
        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            currentPage.setVisible(false);
            currentPage = analysisConfigPage;
            setContentPane(currentPage);
            revalidate();
            repaint();
        }
    }

    private void performAnalysis() {

        // Get selected models
        List<String> selectedModels = new ArrayList<>();
        selectedModels.add("Naive Bayes");
        for (JCheckBox checkBox : modelCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedModels.add(checkBox.getText());
            }
        }

        // Get selected techniques
        List<String> selectedTechniques = new ArrayList<>();
        selectedTechniques.add("no technique");
        for (JCheckBox checkBox : techniqueCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedTechniques.add(checkBox.getText());
            }
        }
        
        // Get selected number of iteractions
        Object value = iterationsField.getValue();
        if (value instanceof Number) {
            numberOfIterations = ((Number) value).intValue();
        }
        else {
            numberOfIterations = 10;
        }

        // Get selected parameters
        List<String> selectedParameters = new ArrayList<>();
        for (JCheckBox checkBox : parameterCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedParameters.add(checkBox.getText());
            }
        }
        if (selectedParameters.isEmpty()){
            selectedParameters.add("F1-Score");
        }
        String parameters = String.join(",", selectedParameters);

        // Get selected file
        File selectedFile = fileChooser.getSelectedFile();
        
        try {
            pythonOutput.append("technique,model," + parameters).append("\n");
            for (String technique : selectedTechniques) {
                for (String model : selectedModels) {
                    for (int i = 0; i < numberOfIterations; i++) {

                        // Execute the Python script passing the CSV file path
                        ProcessBuilder pb = new ProcessBuilder("python", "code\\program_analysis.py", selectedFile.getAbsolutePath(), technique, model, selectedOptimization, selectedCrossValidation, parameters);
                        pb.redirectErrorStream(true);
                        Process process = pb.start();

                        // Read the output from the Python script
                        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                        String line;
                        while ((line = reader.readLine()) != null) {
                            pythonOutput.append(line).append("\n");
                        }

                        // Wait for the process to finish
                        int exitCode = process.waitFor();
                        if (exitCode != 0) {
                            JOptionPane.showMessageDialog(this, "Error on the analysis script", "Error", JOptionPane.ERROR_MESSAGE);
                            //System.exit(0);
                        }
                    }
                }
            }
            
            String returnedString = pythonOutput.toString();

            // Save the returned string as a CSV file
            saveStringAsCSV(returnedString, "results table\\results.csv");
             
             revalidate();
             repaint();

             // Generate image
             ProcessBuilder pb = new ProcessBuilder("python", "code\\program_plot.py");
             pb.redirectErrorStream(true);
             Process process = pb.start();

             int exitCode = process.waitFor();
            if (exitCode != 0) {
                JOptionPane.showMessageDialog(this, "Error on the plot script", "Error", JOptionPane.ERROR_MESSAGE);
            }

            // Show returned image
            ImageIcon returnedImage = new ImageIcon("results image\\graphs.png");
            JLabel imageLabel = new JLabel(returnedImage);
            imageLabel.setIcon(returnedImage);
            JScrollPane scrollPane = new JScrollPane(imageLabel);
            setContentPane(scrollPane);

            pack();
            setLocationRelativeTo(null);
            setVisible(true);

            // Revalidate and repaint the frame to reflect the changes
            revalidate();
            repaint();

        } catch (IOException | InterruptedException ex) {
            ex.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error running Python script: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            System.exit(0);
        }
    }

    private static void saveStringAsCSV(String content, String filePath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(content);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Best_Classification_AI::new);
    }
}
