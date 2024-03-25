import javax.swing.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.Arrays;
import java.util.List;

public class Best_Classification_AI extends JFrame implements ActionListener {
    private JFileChooser fileChooser;
    private JLabel backgroundLabel;
    private JLabel loadingLabel;
    List<String> techniques = Arrays.asList("no technique", "PCA", "IncPCA", "ICA", "LDA", "SMOTE");
    List<String> models = Arrays.asList("Naive Bayes", "SVM", "MLP", "Tree", "KNN", "LogReg");
    StringBuilder pythonOutput = new StringBuilder();

    public Best_Classification_AI() {
        super("Best-Classifiation-AI");

        // Set look and feel to system default
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }

        String imagePath = "resources\\background.gif";
        File imageFile = new File(imagePath);
        String absolutePath = imageFile.getAbsolutePath();
        ImageIcon backgroundImage = new ImageIcon(absolutePath);
        // Create a JLabel with the image
        backgroundLabel = new JLabel(backgroundImage);
        backgroundLabel.setLayout(new BorderLayout());

        // Create components
        JButton openButton = new JButton("Analyse All");
        openButton.addActionListener(this);
        openButton.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 32));

        // Create two bigger buttons on top
        JPanel topPanel = new JPanel();
        topPanel.setOpaque(false);

        JButton Button1 = new JButton("Analyse Models");
        Button1.addActionListener(this);
        Button1.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 24));
        Button1.setPreferredSize(new Dimension(300, 70));

        JButton Button2 = new JButton("Analyse Techniques");
        Button2.addActionListener(this);
        Button2.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 24));
        Button2.setPreferredSize(new Dimension(300, 70));

        // Change layout to GridBagLayout for more flexibility
        topPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(10, 5, 10, 5);
        topPanel.add(Button1, gbc);

        gbc.gridx = 1;
        topPanel.add(Button2, gbc);

        // Layout components
        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.setOpaque(false);
        mainPanel.setBorder(BorderFactory.createEmptyBorder(150, 312, 270, 312));
        mainPanel.add(topPanel, BorderLayout.NORTH);

        // Decrease the size of the "Analyse All" button
        openButton.setPreferredSize(new Dimension(150, 50));

        // Initialize loadingLabel
        loadingLabel = new JLabel("Loading...");
        loadingLabel.setFont(new Font("Lucida Sans Unicode", Font.BOLD, 42));
        loadingLabel.setForeground(Color.WHITE);
        loadingLabel.setVisible(false);

        // Add loadingLabel to a new panel
        JPanel loadingPanel = new JPanel();
        loadingPanel.setOpaque(false);
        loadingPanel.add(loadingLabel);

        // Add loadingPanel to mainPanel below the existing components
        mainPanel.add(loadingPanel, BorderLayout.SOUTH);

        backgroundLabel.add(mainPanel);

        mainPanel.add(openButton, BorderLayout.CENTER);

        // Set the content pane to the background label
        setContentPane(backgroundLabel);

        // Set file chooser
        fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            public boolean accept(File f) {
                return f.getName().toLowerCase().endsWith(".csv") || f.isDirectory();
            }

            public String getDescription() {
                return "CSV files (*.csv)";
            }
        });

        // Set fixed size for the window
        setSize(1280, 720);
        setResizable(false);


        ImageIcon icon = new ImageIcon("resources\\icon.png");
        setIconImage(icon.getImage());
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setVisible(true);
    }

    public void actionPerformed(ActionEvent e) {    
        if (e.getActionCommand().equals("Analyse All")) {
            openCSV();
        } else if (e.getActionCommand().equals("Analyse Models")) {
            techniques = Arrays.asList("no technique");
            openCSV();
        } else if (e.getActionCommand().equals("Analyse Techniques")) {
            models = Arrays.asList("SVM");
            openCSV();
        }
    }

    private void openCSV() {
        loadingLabel.setVisible(true);

        int returnVal = fileChooser.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            handleCSV(fileChooser.getSelectedFile());
        }
        else {
            loadingLabel.setVisible(false);
        }
    }

    private void handleCSV(File file) {
        try {
            pythonOutput.append("technique,model,f1_score,processing_time,memory_usage").append("\n");
            for (String technique : techniques) {
                for (String model : models) {
                    // Execute the Python script passing the CSV file path
                    ProcessBuilder pb = new ProcessBuilder("python", "code\\program_analysis.py", file.getAbsolutePath(), technique, model);
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