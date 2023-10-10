import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class Main {

    public enum Personagens {
        HOMER, BART, LISA, MARGE, MAGGIE, NELSON
    }


    public static void main(String[] args) throws FileNotFoundException {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        List<Map<String, Integer>> data = loadData("src\\main\\resources\\personagens.csv");
        Map<String, Integer> example = new HashMap<>();
        example.put("idade", 13);
        example.put("altura", 158);
        example.put("peso", 40);
        example.put("sexo", 1);

        DecisionTree decisionTree = new DecisionTree();
        //o segundo parametro é o atributo que ele vai aprender a diferenciar
        decisionTree.fit(data, "personagem");

        Personagens personagem = Personagens.values()[decisionTree.eval(example)];
        System.out.println(personagem);
        System.out.println(decisionTree);
    }

    private static List<Map<String, Integer>> loadData(String path) throws FileNotFoundException {
        List<Map<String, Integer>> data = new ArrayList<>();
        Scanner sc = new Scanner(new File(path));
        List<String> labels = new ArrayList<>(List.of(sc.nextLine().split(",")));
        while (sc.hasNext()) {
            String[] values = sc.nextLine().split(",");
            Map<String, Integer> row = new HashMap<>();
            for (int i = 0; i < labels.size(); i++)
                row.put(labels.get(i), Integer.parseInt(values[i]));
            data.add(row);
        }
        sc.close();
        return data;
    }
}
