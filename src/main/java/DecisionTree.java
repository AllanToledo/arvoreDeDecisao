import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.log;

public class DecisionTree {
    public static class DecisionNode {
        public Integer resultado;
        public DecisionNode isTrue;
        public DecisionNode isFalse;
        public Integer threshold;
        public String key;
        public Integer deepth = 0;

        public Integer eval(Map<String, Integer> example) {
            if (resultado != null) return resultado;
            if (example.get(this.key) >= threshold)
                return isTrue.eval(example);
            return isFalse.eval(example);
        }

    }

    DecisionNode rootNode;

    public double entropy(List<Map<String, Integer>> data, String _class) {
        if (data.size() == 0) return 0;
        Map<Integer, Double> fractions = new HashMap<>();
        Double fraction = 1.0 / data.size();
        for (var example : data) {
            Integer value = example.get(_class);
            fractions.put(value, fractions.getOrDefault(value, 0.0) + fraction);
        }
        return -fractions.values().stream().reduce(0.0, (r, pi) -> r + pi * (log(pi) / log(2)));
    }

    public double gainClass(List<Map<String, Integer>> data, String s, String a) {
        if(s.equals(a)) return 0;
        Map<Integer, Integer> values = new HashMap<>();
        for (var row : data) {
            values.put(row.get(a), values.getOrDefault(row.get(a), 0) + 1);
        }
        double entropyS = entropy(data, s);
        double gain = entropyS;
        for (var value : values.entrySet()) {
            double sizeGreater = values.entrySet().stream().map(e -> (e.getKey() >= value.getKey() ? e.getValue() : 0))
                    .reduce(0, Integer::sum);
            double sizeLess = values.entrySet().stream().map(e -> (e.getKey() < value.getKey() ? e.getValue() : 0))
                    .reduce(0, Integer::sum);

            var valuesGreater = data.stream().filter((row) -> row.get(a) >= value.getKey()).toList();
            var valuesLess = data.stream().filter((row) -> row.get(a) < value.getKey()).toList();
            var gainCalculated = sizeGreater / data.size() * entropy(valuesGreater, s) + sizeLess / data.size() * entropy(valuesLess, s);
            gain = Math.min(gain, gainCalculated);
        }

        return entropyS - gain;
    }

    public Map<Integer, Double> gainAtribute(List<Map<String, Integer>> data, String s, String a) {
        Map<Integer, Integer> values = new HashMap<>();
        for (var row : data) {
            values.put(row.get(a), values.getOrDefault(row.get(a), 0) + 1);
        }
        double entropyS = entropy(data, s);
        Map<Integer, Double> gain = new HashMap<>();
        for (var value : values.entrySet()) {
            double sizeGreater = values.entrySet().stream().map(e -> (e.getKey() >= value.getKey() ? e.getValue() : 0))
                    .reduce(0, Integer::sum);
            double sizeLess = values.entrySet().stream().map(e -> (e.getKey() < value.getKey() ? e.getValue() : 0))
                    .reduce(0, Integer::sum);

            var valuesGreater = data.stream().filter((row) -> row.get(a) >= value.getKey()).toList();
            var valuesLess = data.stream().filter((row) -> row.get(a) < value.getKey()).toList();

            gain.put(value.getKey(), entropyS - (sizeGreater / data.size() * entropy(valuesGreater, s) + sizeLess / data.size() * entropy(valuesLess, s)));
        }
        return gain;
    }

    public void fit(List<Map<String, Integer>> data, String s) {
        rootNode = new DecisionNode();
        fit(data, s, rootNode);
    }

    private void fit(List<Map<String, Integer>> data, String s, DecisionNode node) {
        double entropyS = entropy(data, s);
        //Se a entropia cair para valores muito baixo, ele não divide mais, se torna um nó de resultado
        //ou se a profundidade for muito grande tbm
        if (abs(entropyS) <= 0.01 || node.deepth > 5) {
            node.resultado = data.get(0).get(s);
            return;
        }

        AbstractMap.SimpleEntry<String, Double> maxGain = new AbstractMap.SimpleEntry<>("", -1.0);

        //Pega as colunas, ou seja o nome dos atributos
        List<String> columns = new ArrayList<>(data.get(0).keySet().stream().toList());
        Collections.shuffle(columns); //A cada execução do algoritmo alterna entre os atributos com mesmo ganho

        for (var column : columns)  {
            double gain = gainClass(data, s, column);
            if (maxGain.getValue() < gain) {
                maxGain = new AbstractMap.SimpleEntry<>(column, gain); //Salva o atributo com maior ganho
            }
        }

        //Obtem o valor do atributo que obtem o maior ganho
        //<Valor, Ganho>
        Map.Entry<Integer, Double> selectedValue = gainAtribute(data, s, maxGain.getKey())
                .entrySet().stream().max(Map.Entry.comparingByValue()).orElseThrow();

        node.threshold = selectedValue.getKey();
        node.key = maxGain.getKey();
        node.isTrue = new DecisionNode();
        node.isFalse = new DecisionNode();
        node.isTrue.deepth = node.deepth + 1;
        node.isFalse.deepth = node.deepth + 1;

        //Continua a definir o restante da arvore
        fit(data.stream().filter(row -> row.get(node.key) >= node.threshold).toList(), s, node.isTrue);
        fit(data.stream().filter(row -> row.get(node.key) < node.threshold).toList(), s, node.isFalse);

    }

    public Integer eval(Map<String, Integer> example) {
        if (rootNode == null) return null;
        return rootNode.eval(example);
    }

    @Override
    public String toString(){
        return toString(rootNode, "");
    }

    String toString(DecisionNode node, String identation) {
        StringBuilder sb = new StringBuilder();
        if(node.resultado != null) {
            sb.append(identation).append("classe ").append(node.resultado);
            return sb.toString();
        }
        String nextIdentation = identation + "    ";
        sb.append(identation).append(node.key).append(" >= ").append(node.threshold).append(": \n");
        sb.append(toString(node.isTrue, nextIdentation)).append("\n");
        sb.append(identation).append(node.key).append(" < ").append(node.threshold).append(": \n");
        sb.append(toString(node.isFalse, nextIdentation));
        return sb.toString();
    }

}
