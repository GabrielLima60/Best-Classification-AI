import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Best_Classification_AI extends JFrame implements ActionListener {
    private JFileChooser fileChooser;
    private JPanel currentPage;
    private JPanel csvSelectionPage;
    private JPanel analysisConfigPage;
    private JButton selectCSVButton;
    private JButton analyzeButton;
    private List<JCheckBox> modelCheckBoxes;
    private List<JCheckBox> techniqueCheckBoxes;
    private JLabel techniquesLabel;
    private JLabel modelsLabel;
    private StringBuilder pythonOutput = new StringBuilder();

    private List<String> techniques = Arrays.asList("PCA", "IncPCA", "ICA", "LDA");
    private List<String> models = Arrays.asList("SVM", "MLP", "Tree", "KNN", "LogReg");

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
        configPanel.setBorder(BorderFactory.createEmptyBorder(60, 20, 300, 20));

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

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == selectCSVButton) {
            openCSVSelection();
        } else if (e.getSource() == analyzeButton) {
            performAnalysis();
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
        
        List<String> selectedModels = new ArrayList<>();
        selectedModels.add("Naive Bayes");
        for (JCheckBox checkBox : modelCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedModels.add(checkBox.getText());
            }
        }

        // Get selected technique checkboxes
        List<String> selectedTechniques = new ArrayList<>();
        selectedTechniques.add("no technique");
        for (JCheckBox checkBox : techniqueCheckBoxes) {
            if (checkBox.isSelected()) {
                selectedTechniques.add(checkBox.getText());
            }
        }
        File selectedFile = fileChooser.getSelectedFile();
        
        try {
            pythonOutput.append("technique,model,f1_score,processing_time,memory_usage").append("\n");
            for (String technique : selectedTechniques) {
                for (String model : selectedModels) {
                    // Execute the Python script passing the CSV file path
                    ProcessBuilder pb = new ProcessBuilder("python", "code\\program_analysis.py", selectedFile.getAbsolutePath(), technique, model);
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
                        JOptionPane.showMessageDialog(this, "Error analysis script", "Error", JOptionPane.ERROR_MESSAGE);
                        System.exit(0);
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
                JOptionPane.showMessageDialog(this, "Error plot script", "Error", JOptionPane.ERROR_MESSAGE);
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
