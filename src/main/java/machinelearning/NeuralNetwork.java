package machinelearning;

import org.ejml.simple.SimpleMatrix;

import java.util.*;
import java.util.stream.Collectors;

public class NeuralNetwork {

    SimpleMatrix inputs;
    SimpleMatrix weights;
    SimpleMatrix thresholds;

    Integer classesSize;

    String _class;
    public double learningRate = 0.1;

    public int eval(Map<String, Double> e) {
        //Obtem os resultados ja aplicando a funçãod e ativação
        SimpleMatrix results = applyActivactionFunction(processInputs(e));

        //Salva o indice do neuronio que possui maior valor após o processamento da entrada
        //Como o neuronio 0 representa a classe 0 e assim por diante, basta saber o indice do neuronio
        //que sabemos qual foi a classe indicada pela rede neural

        double classProbabilty = Double.NEGATIVE_INFINITY;
        int selectedClass = -1;
        for (int i = 0; i < classesSize; i++)
            if (results.get(0, i) > classProbabilty) {
                selectedClass = i;
                classProbabilty = results.get(0, i);
            }

        return selectedClass;
    }

    public void build(List<Map<String, Double>> data, String _class) {
        this._class = _class;
        Set<Double> classList = data.stream().map((e) -> e.get(_class)).collect(Collectors.toSet());
        this.classesSize = classList.size();
        Set<String> attributesList = data.get(0).keySet();
        int attributesSize = attributesList.size() - 1;
        //Cria uma matriz para entradas [x1, x2, ..., xn]
        this.inputs = new SimpleMatrix(1, attributesSize);

        //Cria uma matriz para os pesos, cada coluna indica um neuronio de saida
        //Cada linha indica um atributo
        // [w11, w12, ..., w1j]
        // [w21, w22, ..., w2j]
        // [..., ..., ..., ...]
        // [wi1, wi2, ..., wij]
        this.weights = new SimpleMatrix(attributesSize, classesSize);
        weights.fill(1); //Valores iniciais

        //Cria uma matriz para os valores que serão o limiar da saida
        //1 linha e N colunas, onde N é o número de classes
        // [O1, O2, ..., On]
        this.thresholds = new SimpleMatrix(1, classesSize);
        thresholds.fill(0);
    }

    public void fit(List<Map<String, Double>> data, Integer epochs) {
        for (int i = 0; i < epochs; i++) {

            //A cada sessão de treinamento aleatoriza a entrada, para evitar viciar
            // numa classe e dificultar o aprendizado
            Collections.shuffle(data);
            for (var e : data) {
                fit(e);
            }
        }
    }

    public void fit(Map<String, Double> e) {
        //Importante ter entendido bem como funciona o método build()

        //Para calcular o limiar, nós precisamos dos valores antes da aplicação da função de ativação,
        //por isso eu armazeno em variaveis diferentes
        SimpleMatrix processedInputs = processInputs(e);
        SimpleMatrix neuronResults = applyActivactionFunction(processedInputs);

        //Para evitar usar muitos loops para realizar os calculos, vou abusar da biblioteca de matrizes

        SimpleMatrix inputsForCorrection = new SimpleMatrix(weights.getNumRows(), weights.getNumCols());
        SimpleMatrix expectedNeuronValues = new SimpleMatrix(weights.getNumRows(), weights.getNumCols());
        SimpleMatrix actualNeuronValues = new SimpleMatrix(weights.getNumRows(), weights.getNumCols());

        for (int i = 0; i < weights.getNumRows(); i++)
            for (int j = 0; j < weights.getNumCols(); j++) {
                double expectedNeuronValue = e.get(_class) == j ? 1.0 : 0.0;
                //Cria uma matriz replicando os valores esperados
                //Se era esperado a saida [0, 1, 0] ele cria uma matriz
                //[0, 1, 0]
                //[0, 1, 0]
                //[0, 1, 0]
                //[0, 1, 0]
                //Isso por que vamos treinar todos os pesos numa chamada de método apenas
                expectedNeuronValues.set(i, j, expectedNeuronValue);
                //Mesma ideia acima, porém usando os valores obtidos pelo processamento da rede
                actualNeuronValues.set(i, j, neuronResults.get(0, j));
                //Nesse caso preenche os valores de entradas verticalmente
                //Para entrada [1, 2, 3, 4] por exemplo, fica:
                //[1, 1, 1]
                //[2, 2, 2]
                //[3, 3, 3]
                //[4, 4, 4]
                inputsForCorrection.set(i, j, inputs.get(0, i));
            }

        //Aqui que acontece o treinamento de todos os pesos
        //Ele calcula seguinte expressão
        //Onde i = linha, j = coluna
        //wij = wij + (input_i) * (learningRate) * (expected_ij - actual_ij)
        weights = weights.plus(
                inputsForCorrection.scale(learningRate).elementMult(
                        expectedNeuronValues.minus(actualNeuronValues)));

        //Aqui acontece a mesma coisa para o ajuste do limiar
        expectedNeuronValues = new SimpleMatrix(1, weights.getNumCols());

        for (int j = 0; j < weights.getNumCols(); j++)
            expectedNeuronValues.set(0, j, e.get(_class) == j ? 1.0 : 0.0);

        //Aqui funciona de maneira muito similiar, porém o input_i é a entrada para a função de ativação
        //Ou seja, os valores processados sem a aplicação da função de ativação
        //Oij = Oij + (sum_i) * (learningRate) * (expected_ij - actual_ij)
        thresholds = thresholds.plus(
                processedInputs.scale(learningRate).elementMult(
                        expectedNeuronValues.minus(neuronResults)));

    }

    public SimpleMatrix processInputs(Map<String, Double> e) {
        int i = 0;
        //Transfere os valores para os neuronios de entrada
        for(Map.Entry<String, Double> entry: e.entrySet()){
            //Evita colocar o atributo personagem como valor de entrada, causaria um erro
            if(entry.getKey().equals(_class)) continue;
            inputs.set(0, i++, entry.getValue());
        }
        //Calcula os pesos já aplicando a somatória, pois é uma multiplicação de matrizes
        //Depois subtrai o limiar, mas ainda não aplica a função de ativação
        return inputs.mult(weights).minus(thresholds);
    }

    public SimpleMatrix applyActivactionFunction(SimpleMatrix processedInputs) {
        for (int i = 0; i < classesSize; i++) {
            double x = processedInputs.get(0, i);
            //Aplica a função de ativação (Sigmoid) para cada resultado veja:
            //https://www.deeplearningbook.com.br/funcao-de-ativacao/#:~:text=%C3%A9%20altamente%20desejada.-,Sigm%C3%B3ide,-Sigm%C3%B3ide%20%C3%A9%20uma
            processedInputs.set(0, i,  (1 / (1 + Math.exp(-x))));
        }
        return processedInputs;
    }

    @Override
    public String toString() {
        //Função simples para imprimir os valores
        return "INPUTS\n" + inputs + '\n' +
                "WEIGHTS\n" + weights + '\n' +
                "THRESHOLDS\n" + thresholds + '\n';
    }
}
