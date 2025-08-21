import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Principal {

    private static final double TAXA_APRENDIZADO = 0.1;

    public static void main(String[] args) throws Exception {

        System.out.println("Iniciando leitura do arquivo CSV...");
        ArrayList<double[]> dados = lerArquivo("src/dados.csv");
        if (dados.isEmpty()) {
            System.out.println("Nenhum dado foi lido do arquivo.");
            return;
        } else {
            System.out.println("Dados lidos com sucesso. Total de registros: " + dados.size());
        }

        // Inicializa sinapses
        double[] vetorW1 = new double[17];
        double[] vetorW2 = new double[17];
        double[] vetorW3 = new double[3]; 
        double[] vetorW4 = new double[3]; 

        vetorW1 = inicializarSinapse(vetorW1);
        vetorW2 = inicializarSinapse(vetorW2);
        vetorW3 = inicializarSinapse(vetorW3);
        vetorW4 = inicializarSinapse(vetorW4);

        double[][] sinapses = {vetorW1, vetorW2, vetorW3, vetorW4};
        System.out.println("Sinapses inicializadas com sucesso.");

        // Treina a rede neural
        double erroDesejado = 0.01;
        int maxEpocas = 100000;
        treinarRede(dados, sinapses, erroDesejado, maxEpocas);
    }


    public static void treinarRede(ArrayList<double[]> dados, double[][] sinapses, double erroDesejado, int maxEpocas) {
        int epocaAtual = 0;
        double erroMedio = 1.0;

        while (erroMedio > erroDesejado && epocaAtual < maxEpocas) {
            double somaErroEpoca = 0;

            for (int i = 0; i < dados.size(); i++) {
                somaErroEpoca += rodarAmostra(dados.get(i), sinapses);
            }

            erroMedio = somaErroEpoca / dados.size();
            System.out.println("Época " + epocaAtual + " - Erro médio: " + erroMedio);
            epocaAtual++;
        }

        if (erroMedio <= erroDesejado) {
            System.out.println("Treinamento finalizado na época " + (epocaAtual - 1) + " (erro médio atingiu " + erroMedio + ").");
        } else {
            System.out.println("Treinamento atingiu o limite de " + maxEpocas + " épocas. Erro final: " + erroMedio);
        }

        // Salva as sinapses finais
        salvarSinapses("src/sinapses.csv", sinapses);
        System.out.println("Sinapses finais salvas com sucesso.");
    }

    public static double rodarAmostra(double[] entrada, double[][] sinapses) {
        // Extrai os pesos da matriz
        double[] vetorW1 = sinapses[0];
        double[] vetorW2 = sinapses[1];
        double[] vetorW3 = sinapses[2];
        double[] vetorW4 = sinapses[3];

        // Camada oculta
        double v1 = calcularV(entrada, vetorW1);
        double v2 = calcularV(entrada, vetorW2);
        double y1 = calcularSaida(v1);
        double y2 = calcularSaida(v2);

        // Camada de saída
        double[] entradaSaida = {1.0, y1, y2};
        double v3 = calcularV(entradaSaida, vetorW3);
        double v4 = calcularV(entradaSaida, vetorW4);
        double y3 = calcularSaida(v3);
        double y4 = calcularSaida(v4);

        // Erros na saída
        double e3 = calcularErro(y3, entrada[17]);
        double e4 = calcularErro(y4, entrada[18]);

        // Gradientes da camada de saída
        double s3 = calcularGradiente(e3, y3);
        double s4 = calcularGradiente(e4, y4);

        // Gradientes da camada oculta
        double s1 = retropropagar(y1, s3, s4, vetorW3[1], vetorW4[1]);
        double s2 = retropropagar(y2, s3, s4, vetorW3[2], vetorW4[2]);

        // Atualiza os pesos da camada de saída
        atualizarNeuronios(s3, entradaSaida, vetorW3);
        atualizarNeuronios(s4, entradaSaida, vetorW4);

        // Atualiza os pesos da camada oculta
        atualizarNeuronios(s1, entrada, vetorW1);
        atualizarNeuronios(s2, entrada, vetorW2);

        // Retorna o erro quadrático médio
        return (e3 * e3 + e4 * e4) / 2.0;
    }

    public static double[] inicializarSinapse(double[] vetorSinapses) {
        Random random = new Random();
        for(int i = 0; i < vetorSinapses.length; i++){
            vetorSinapses[i] = -1 + 2 * random.nextDouble(); // Remove o cast para (int)
        }
        return vetorSinapses;
    }

    public static double calcularV(double[] vetorX, double[] vetorW) {
        double v = 0;
        for(int i = 0; i < vetorW.length; i++) {
            v += vetorX[i] * vetorW[i];
        }
        return v;
    }

    public static double calcularSaida(double v) {
        return 1.0 / (1.0 + Math.exp(-TAXA_APRENDIZADO * v));
    }

    public static double calcularErro(double y, double yD) {
        return yD - y;
    }

    public static double calcularGradiente(double e, double y) {
        return e * TAXA_APRENDIZADO * y * (1 - y);
    }

    public static void atualizarNeuronios(double s, double[] entrada, double[] vetorW) {
        for(int i = 0; i < vetorW.length; i++){
            vetorW[i] = vetorW[i] + TAXA_APRENDIZADO * s * entrada[i];
        }
    }

    public static double retropropagar(double y, double s1, double s2, double w1, double w2) {
        double erroPropagado = (s1 * w1) + (s2 * w2);
        return TAXA_APRENDIZADO*y*(1-y)*erroPropagado;
    }

    public static ArrayList<double[]> lerArquivo(String caminho) {
        ArrayList<double[]> dados = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(caminho))) {
            String linha;
            while ((linha = br.readLine()) != null) {
                String[] valores = linha.split(",");

                if (valores.length != 19) { 
                    System.err.println("Linha inválida (esperado 19 valores): " + linha);
                    continue;
                }

                double[] vetor = new double[19];

                for (int i = 0; i < 17; i++) {
                    vetor[i] = Double.parseDouble(valores[i]);
                }

                vetor[17] = Double.parseDouble(valores[17]); 
                vetor[18] = Double.parseDouble(valores[18]); 

                dados.add(vetor);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return dados;
    }

    public static void salvarSinapses(String caminho, double[][] vetores) {
        try (FileWriter writer = new FileWriter(caminho)) {
        
            // Cada vetor vai ser uma linha
            for (double[] vetor : vetores) {
                for (int i = 0; i < vetor.length; i++) {
                    writer.append(String.valueOf(vetor[i]));
                    if (i < vetor.length - 1) {
                        writer.append(",");
                    }
                }
                writer.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double[][] lerSinapsesCSV(String caminho) {
        List<double[]> listaVetores = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(caminho))) {
            String linha;
            while ((linha = br.readLine()) != null) {
                String[] valores = linha.split(",");
                double[] vetor = new double[valores.length];
                for (int i = 0; i < valores.length; i++) {
                    vetor[i] = Double.parseDouble(valores[i]);
                }
                listaVetores.add(vetor);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Converte a lista em array bidimensional
        return listaVetores.toArray(new double[0][]);
    }
}