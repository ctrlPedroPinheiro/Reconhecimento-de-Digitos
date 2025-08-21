import java.io.BufferedReader;
import java.io.File; 
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner; 

public class Principal {

    private static final double TAXA_APRENDIZADO = 0.1;
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int opcao = 0;

        while (opcao != 3) {
            System.out.println("\n--- MENU DA REDE NEURAL ---");
            System.out.println("1. Treinar a Rede");
            System.out.println("2. Testar a Rede");
            System.out.println("3. Sair");
            System.out.print("Escolha uma opção: ");

            try {
                opcao = scanner.nextInt();

                switch (opcao) {
                    case 1:
                        iniciarTreinamento();
                        break;
                    case 2:
                        testarRede();
                        break;
                    case 3:
                        System.out.println("Saindo do programa...");
                        break;
                    default:
                        System.out.println("Opção inválida. Tente novamente.");
                        break;
                }
            } catch (Exception e) {
                System.err.println("Entrada inválida. Por favor, digite um número.");
                scanner.next(); 
            }
        }
        scanner.close();
    }


    public static void iniciarTreinamento() {
        System.out.println("\n--- INICIANDO TREINAMENTO ---");
        
        // Lê os dados do arquivo de treinamento
        ArrayList<double[]> dados = lerArquivo("src/dados.csv");
        if (dados.isEmpty()) {
            System.out.println("Nenhum dado de treinamento encontrado.");
            return;
        }
        System.out.println("Dados de treinamento lidos com sucesso. Total de registros: " + dados.size());

        // Inicializa sinapses novas
        double[] vetorW1 = inicializarSinapse(new double[17]);
        double[] vetorW2 = inicializarSinapse(new double[17]);
        double[] vetorW3 = inicializarSinapse(new double[3]);
        double[] vetorW4 = inicializarSinapse(new double[3]);
        double[][] sinapses = {vetorW1, vetorW2, vetorW3, vetorW4};
        System.out.println("Sinapses novas foram inicializadas.");

        // Parâmetros do treinamento
        double erroDesejado = 0.01;
        
        // Chama a função principal de treinamento
        treinarRede(dados, sinapses, erroDesejado);
    }

    public static void testarRede() {
        System.out.println("\n--- INICIANDO TESTE ---");

        File arquivoSinapses = new File("src/sinapses.csv");
        if (!arquivoSinapses.exists()) {
            System.out.println("ERRO: Arquivo 'sinapses.csv' não encontrado.");
            System.out.println("Você precisa treinar a rede primeiro (Opção 1).");
            return;
        }

        double[][] sinapses = lerSinapsesCSV("src/sinapses.csv");
        System.out.println("Sinapses treinadas carregadas com sucesso.");

        ArrayList<double[]> dadosTeste = lerArquivo("src/testes.csv"); 
        if (dadosTeste.isEmpty()) {
            System.out.println("Nenhum dado de teste encontrado.");
            return;
        }

        double somaErroTotal = 0;
        int acertos = 0; // NOVO: Contador para as previsões corretas

        for (int i = 0; i < dadosTeste.size(); i++) {
            double[] entrada = dadosTeste.get(i);
        
            
            double v1 = calcularV(entrada, sinapses[0]);
            double v2 = calcularV(entrada, sinapses[1]);
            double y1 = calcularSaida(v1);
            double y2 = calcularSaida(v2);

            double[] entradaSaida = {1.0, y1, y2};
            double v3 = calcularV(entradaSaida, sinapses[2]);
            double v4 = calcularV(entradaSaida, sinapses[3]);

            // Saídas brutas
            double y3_bruto = calcularSaida(v3);
            double y4_bruto = calcularSaida(v4);

        
            int previsao1 = (y3_bruto >= 0.5) ? 1 : 0;
            int previsao2 = (y4_bruto >= 0.5) ? 1 : 0;

            // Saídas desejadas
            int esperado1 = (int) entrada[17];
            int esperado2 = (int) entrada[18];

        
            String resultado = "INCORRETO";
            if (previsao1 == esperado1 && previsao2 == esperado2) {
                acertos++;
                resultado = "CORRETO";
            }

            // Calcula o erro para estatísticas
            double erro1 = esperado1 - y3_bruto;
            double erro2 = esperado2 - y4_bruto;
            somaErroTotal += (erro1 * erro1 + erro2 * erro2) / 2.0;

            System.out.printf("Amostra #%d: Previsto [%d, %d] (Bruto: [%.4f, %.4f]) | Esperado [%d, %d] -> %s\n",
                i + 1, previsao1, previsao2, y3_bruto, y4_bruto, esperado1, esperado2, resultado);
        }

    
        double precisao = (double) acertos / dadosTeste.size() * 100.0;
        double erroMedioTeste = somaErroTotal / dadosTeste.size();
    
        System.out.println("\n--- RESULTADO DO TESTE ---");
        System.out.printf("Previsões Corretas: %d de %d\n", acertos, dadosTeste.size());
        System.out.printf("Precisão: %.2f%%\n", precisao);
        System.out.printf("Erro Médio Quadrático Final: %.6f\n", erroMedioTeste);
    }
    
    public static void treinarRede(ArrayList<double[]> dados, double[][] sinapses, double erroDesejado) {
        int epocaAtual = 0;
        double erroMedio = 1.0;

        while (erroMedio > erroDesejado) {
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
        } 
        salvarSinapses("src/sinapses.csv", sinapses);
        System.out.println("Sinapses finais salvas com sucesso em 'src/sinapses.csv'.");
    }

    public static double rodarAmostra(double[] entrada, double[][] sinapses) {
        double[] vetorW1 = sinapses[0];
        double[] vetorW2 = sinapses[1];
        double[] vetorW3 = sinapses[2];
        double[] vetorW4 = sinapses[3];

        double v1 = calcularV(entrada, vetorW1);
        double v2 = calcularV(entrada, vetorW2);
        double y1 = calcularSaida(v1);
        double y2 = calcularSaida(v2);

        double[] entradaSaida = {1.0, y1, y2};
        double v3 = calcularV(entradaSaida, vetorW3);
        double v4 = calcularV(entradaSaida, vetorW4);
        double y3 = calcularSaida(v3);
        double y4 = calcularSaida(v4);

        double e3 = calcularErro(y3, entrada[17]);
        double e4 = calcularErro(y4, entrada[18]);

        double s3 = calcularGradiente(e3, y3);
        double s4 = calcularGradiente(e4, y4);

        double s1 = retropropagar(y1, s3, s4, vetorW3[1], vetorW4[1]);
        double s2 = retropropagar(y2, s3, s4, vetorW3[2], vetorW4[2]);

        atualizarNeuronios(s3, entradaSaida, vetorW3);
        atualizarNeuronios(s4, entradaSaida, vetorW4);
        atualizarNeuronios(s1, entrada, vetorW1);
        atualizarNeuronios(s2, entrada, vetorW2);

        return (e3 * e3 + e4 * e4) / 2.0;
    }
    
    public static double[] inicializarSinapse(double[] vetorSinapses) {
        Random random = new Random();
        for(int i = 0; i < vetorSinapses.length; i++){
            vetorSinapses[i] = -1 + 2 * random.nextDouble();
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

    public static double retropropagar(double y, double s_out1, double s_out2, double w_h_out1, double w_h_out2) {
        double erroPropagado = (s_out1 * w_h_out1) + (s_out2 * w_h_out2);
        return TAXA_APRENDIZADO * y * (1 - y) * erroPropagado;
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
                for (int i = 0; i < 19; i++) {
                    vetor[i] = Double.parseDouble(valores[i]);
                }
                dados.add(vetor);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dados;
    }

    public static void salvarSinapses(String caminho, double[][] vetores) {
        try (FileWriter writer = new FileWriter(caminho)) {
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
        return listaVetores.toArray(new double[0][]);
    }
}