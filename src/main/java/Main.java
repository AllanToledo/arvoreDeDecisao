import machinelearning.DecisionTree;
import machinelearning.NeuralNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.sql.SQLOutput;
import java.util.*;

public class Main {

    public enum Personagens {
        HOMER, BART, LISA, MARGE, MAGGIE, NELSON
    }


    public static void main(String[] args) throws FileNotFoundException {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        //Rede Neural não é capaz de distinguir outros conjuntos de dados, pois não são linearmente separaveis
        //Para isso, teriamos que usar um Multilayer Perceptron, mas isso complicaria bem mais o método de aprendizado
        //pois teria que usar Backpropagation e matemática bem mais avançada, aqui ta bom
        List<Map<String, Double>> data = loadData("src\\main\\resources\\personagens-v1.csv");
        String attributeClass = "personagem";

        List<Map<String, Double>> validations = loadValidationValues();

        DecisionTree decisionTree = new DecisionTree();
        //o segundo parametro é o atributo que ele vai aprender a classificar
        decisionTree.fit(data, attributeClass);
        System.out.println("Estrutura da Arvore de Decisão");
        System.out.println(decisionTree);

        //A rede neural obtem um melhor desempenho aplicando a normalização dos valores entre 0..1
        boolean applicarNormalizacao = true;
        Map<String, Double> maxValues = getMaxValues(data);
        if(applicarNormalizacao) normalize(data, maxValues, attributeClass);

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.build(data, attributeClass);
        neuralNetwork.learningRate = 0.5;
        neuralNetwork.fit(data, 6);
        System.out.println("\nOrdem das entradas: ");
        for(var key: data.get(0).keySet())
            if(!key.equals(attributeClass)) System.out.println(key);

        System.out.println("\nEstrutura de Rede Neural");
        System.out.println(neuralNetwork);

        System.out.println("\nValidação das classes");
        for(var e: validations){
            Personagens personagem;
            personagem = Personagens.values()[e.get("personagem").intValue()];
            System.out.printf("Esperado \tArvoreDecisao \tRedeNeural\n%s", personagem);

            int decisionTreeResult = decisionTree.eval(e);
            personagem = Personagens.values()[decisionTreeResult];
            System.out.printf("\t\t%s", personagem);
            if(applicarNormalizacao) normalize(e, maxValues, attributeClass); //normalizando para a rede neura
            int neuralNetworkResult = neuralNetwork.eval(e);
            personagem = Personagens.values()[neuralNetworkResult];
            System.out.printf("\t\t\t%s\n", personagem);
        }

    }

    private static List<Map<String, Double>> loadValidationValues() {
        List<Map<String, Double>> validations;
        validations = new ArrayList<>();
        //LISA
        HashMap<String, Double> validation = new HashMap<>();
        validation.put("idade", 13.0);
        validation.put("altura", 158.0);
        validation.put("peso", 40.0);
        validation.put("sexo", 1.0);
        validation.put("personagem", 2.0);
        validations.add(validation);

        //BART
        validation = new HashMap<>();
        validation.put("idade", 13.0);
        validation.put("altura", 158.0);
        validation.put("peso", 40.0);
        validation.put("sexo", 0.0);
        validation.put("personagem", 1.0);
        validations.add(validation);

        //HOMER
        validation = new HashMap<>();
        validation.put("idade", 46.0);
        validation.put("altura", 168.0);
        validation.put("peso", 162.0);
        validation.put("sexo", 0.0);
        validation.put("personagem", 0.0);
        validations.add(validation);
        return validations;
    }

    private static Map<String, Double> getMaxValues(List<Map<String, Double>> data) {
        return data.stream().reduce(new HashMap<>(), (e, r) -> {
            for (var prop : r.entrySet()) {
                e.put(prop.getKey(), Math.max(prop.getValue(), e.getOrDefault(prop.getKey(), 0.0)));
            }
            return e;
        });
    }

    private static void normalize(List<Map<String, Double>> data,
                                  Map<String, Double> maxValues,
                                  String exclude) {
        for (var e : data) normalize(e, maxValues, exclude);
    }

    private static void normalize(Map<String, Double> e, Map<String, Double> maxValues, String exclude) {
        for (String key : e.keySet()) {
            if (key.equals(exclude)) continue;
            e.put(key, e.get(key) / maxValues.get(key));
        }
    }

    private static List<Map<String, Double>> loadData(String path) throws FileNotFoundException {
        List<Map<String, Double>> data = new ArrayList<>();
        Scanner sc = new Scanner(new File(path));
        List<String> labels = new ArrayList<>(List.of(sc.nextLine().split(",")));
        while (sc.hasNext()) {
            String[] values = sc.nextLine().split(",");
            Map<String, Double> row = new HashMap<>();
            for (int i = 0; i < labels.size(); i++)
                row.put(labels.get(i), Double.parseDouble(values[i]));
            data.add(row);
        }
        sc.close();
        return data;
    }
}
