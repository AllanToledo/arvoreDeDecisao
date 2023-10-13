package machinelearning;

import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.log;

public class DecisionTree {
    public static class DecisionNode {
        public Integer result;
        public DecisionNode isTrue;
        public DecisionNode isFalse;
        public Double threshold;
        public String attribute;
        public Double depth = 0.0;

        public Integer eval(Map<String, Double> example) {
            if (result != null) return result;
            if (example.get(attribute) >= threshold)
                return isTrue.eval(example);
            return isFalse.eval(example);
        }

    }

    DecisionNode rootNode;

    public double entropy(List<Map<String, Double>> data, String _class) {
        if (data.size() == 0.0) return 0.0;
        Map<Double, Double> fractions = new HashMap<>();
        Double fraction = 1.00 / data.size();
        for (var example : data) {
            Double value = example.get(_class);
            fractions.put(value, fractions.getOrDefault(value, 0.00) + fraction);
        }
        return -fractions.values().stream().reduce(0.00, (r, pi) -> r + pi * (log(pi) / log(2.0)));
    }

    public double gainClass(List<Map<String, Double>> data, String s, String a) {
        if(s.equals(a)) return 0.0;
        Map<Double, Double> values = new HashMap<>();
        for (var row : data) {
            values.put(row.get(a), values.getOrDefault(row.get(a), 0.0) + 1.0);
        }
        double entropyS = entropy(data, s);
        double gain = entropyS;
        for (var value : values.entrySet()) {
            double sizeGreater = values.entrySet().stream().map(e -> (e.getKey() >= value.getKey() ? e.getValue() : 0.0))
                    .reduce(0.0, Double::sum);
            double sizeLess = values.entrySet().stream().map(e -> (e.getKey() < value.getKey() ? e.getValue() : 0.0))
                    .reduce(0.0, Double::sum);

            var valuesGreater = data.stream().filter((row) -> row.get(a) >= value.getKey()).toList();
            var valuesLess = data.stream().filter((row) -> row.get(a) < value.getKey()).toList();
            var gainCalculated = sizeGreater / data.size() * entropy(valuesGreater, s) + sizeLess / data.size() * entropy(valuesLess, s);
            gain = Math.min(gain, gainCalculated);
        }

        return entropyS - gain;
    }

    public Map<Double, Double> gainAtribute(List<Map<String, Double>> data, String s, String a) {
        Map<Double, Double> values = new HashMap<>();
        for (var row : data) {
            values.put(row.get(a), values.getOrDefault(row.get(a), 0.0) + 1.0);
        }
        double entropyS = entropy(data, s);
        Map<Double, Double> gain = new HashMap<>();
        for (var value : values.entrySet()) {
            double sizeGreater = values.entrySet().stream().map(e -> (e.getKey() >= value.getKey() ? e.getValue() : 0.0))
                    .reduce(0.0, Double::sum);
            double sizeLess = values.entrySet().stream().map(e -> (e.getKey() < value.getKey() ? e.getValue() : 0.0))
                    .reduce(0.0, Double::sum);

            var valuesGreater = data.stream().filter((row) -> row.get(a) >= value.getKey()).toList();
            var valuesLess = data.stream().filter((row) -> row.get(a) < value.getKey()).toList();

            gain.put(value.getKey(), entropyS - (sizeGreater / data.size() * entropy(valuesGreater, s) + sizeLess / data.size() * entropy(valuesLess, s)));
        }
        return gain;
    }

    public void fit(List<Map<String, Double>> data, String _class) {
        rootNode = new DecisionNode();
        fit(data, _class, rootNode);
    }

    private void fit(List<Map<String, Double>> data, String s, DecisionNode node) {
        double entropyS = entropy(data, s);
        //Se a entropia cair para valores muito baixo, ele não divide mais, se torna um nó de resultado
        //ou se a profundidade for muito grande tbm
        if (abs(entropyS) <= 0.010 || node.depth > 5.0) {
            node.result = data.get(0).get(s).intValue();
            return;
        }

        AbstractMap.SimpleEntry<String, Double> maxGain = new AbstractMap.SimpleEntry<>("", -1.00);

        //Pega as colunas, ou seja o nome dos atributos
        List<String> columns = new ArrayList<>(data.get(0).keySet().stream().toList());
        Collections.shuffle(columns); //A cada execução do algoritmo alterna entre os atributos com mesmo ganho

        for (var column : columns)  {
            if(column.equals(s)) continue;
            double gain = gainClass(data, s, column);
            if (maxGain.getValue() < gain) {
                maxGain = new AbstractMap.SimpleEntry<>(column, gain); //Salva o atributo com maior ganho
            }
        }

        //Obtem o valor do atributo que obtem o maior ganho
        //<Valor, Ganho>
        Map.Entry<Double, Double> selectedValue = gainAtribute(data, s, maxGain.getKey())
                .entrySet().stream().max(Map.Entry.comparingByValue()).orElseThrow();

        node.threshold = selectedValue.getKey();
        node.attribute = maxGain.getKey();
        node.isTrue = new DecisionNode();
        node.isFalse = new DecisionNode();
        node.isTrue.depth = node.depth + 1.0;
        node.isFalse.depth = node.depth + 1.0;

        //Continua a definir o restante da arvore
        fit(data.stream().filter(row -> row.get(node.attribute) >= node.threshold).toList(), s, node.isTrue);
        fit(data.stream().filter(row -> row.get(node.attribute) < node.threshold).toList(), s, node.isFalse);

    }

    public Integer eval(Map<String, Double> example) {
        if (rootNode == null) return null;
        return rootNode.eval(example);
    }

    @Override
    public String toString(){
        return toString(rootNode, "");
    }

    String toString(DecisionNode node, String identation) {
        StringBuilder sb = new StringBuilder();
        if(node.result != null) {
            sb.append(identation).append("classe ").append(node.result);
            return sb.toString();
        }
        String nextIdentation = identation + "    ";
        sb.append(identation).append(node.attribute).append(" >= ").append(node.threshold).append(": \n");
        sb.append(toString(node.isTrue, nextIdentation)).append("\n");
        sb.append(identation).append(node.attribute).append(" < ").append(node.threshold).append(": \n");
        sb.append(toString(node.isFalse, nextIdentation));
        return sb.toString();
    }

}
