import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class KNN {
    static class DataPoint {
        String label;
        double[] features;

        DataPoint(String label, double[] features) {
            this.label = label;
            this.features = features;
        }
    }

    // Load CSV file
    public static List<DataPoint> loadFile(String filename) throws IOException {
        List<DataPoint> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] features = Arrays.stream(values, 0, values.length - 1)
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                String label = values[values.length - 1];
                data.add(new DataPoint(label, features));
            }
        }
        return data;
    }

    // Calculate Euclidean distance
    public static double euclideanDistance(double[] v1, double[] v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.length; i++) {
            sum += Math.pow(v1[i] - v2[i], 2);
        }
        return Math.sqrt(sum);
    }

    // KNN classification
    public static String knnClassify(List<DataPoint> trainSet, double[] testVector, int k) {
        List<Map.Entry<Double, String>> distances = new ArrayList<>();
        for (DataPoint dp : trainSet) {
            double dist = euclideanDistance(dp.features, testVector);
            distances.add(new AbstractMap.SimpleEntry<>(dist, dp.label));
        }

        // Sort by distance
        distances.sort(Comparator.comparingDouble(Map.Entry::getKey));

        // Get k nearest neighbors
        Map<String, Integer> labelCount = new HashMap<>();
        for (int i = 0; i < k; i++) {
            String label = distances.get(i).getValue();
            labelCount.put(label, labelCount.getOrDefault(label, 0) + 1);
        }

        // Find most common label
        return Collections.max(labelCount.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Calculate accuracy
    public static double accuracy(List<DataPoint> trainSet, List<DataPoint> testSet, int k) {
        int count = 0;
        for (DataPoint dp : testSet) {
            String predictedLabel = knnClassify(trainSet, dp.features, k);
            if (predictedLabel.equals(dp.label)) {
                count++;
            }
        }
        return (count / (double) testSet.size()) * 100;
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.out.println("Usage: java KNN <k> <train_file> <test_file>");
            return;
        }

        try {
            int k = Integer.parseInt(args[0]);
            String trainFile = args[1];
            String testFile = args[2];

            List<DataPoint> trainSet = loadFile(trainFile);
            List<DataPoint> testSet = loadFile(testFile);

            double acc = accuracy(trainSet, testSet, k);
            System.out.println("The accuracy is: " + acc + "%");

            Scanner scanner = new Scanner(System.in);
            {
                System.out.print("Enter a new feature vector separated by commas: ");
                String input = scanner.nextLine();
                double[] testVector = Arrays.stream(input.split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                String classification = knnClassify(trainSet, testVector, k);
                System.out.println("The class of the feature vector is: " + classification);
            }

        } catch (NumberFormatException e) {
            System.out.println("Please enter numeric values correctly.");
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }


}
