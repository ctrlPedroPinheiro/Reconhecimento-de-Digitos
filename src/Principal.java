import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class Principal {
    public static void main(String[] args) throws Exception {

        // Lê os dados do arquivo CSV
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

        System.out.println("Sinapses inicializadas com sucesso.");

        // Encontrando o valor de V
        double v1 = treinarV(dados.get(0), vetorW1);
        double v2 = treinarV(dados.get(0), vetorW2);

        // Calculando a saída
        double a = 0.1;
        double y1 = calcularSaida(v1, a);
        double y2 = calcularSaida(v2, a);

        // Encontrando o valor de V da camada de saída
        double v3 = treinarV(new double[]{1, y1, y2}, vetorW3);
        double v4 = treinarV(new double[]{1, y1, y2}, vetorW4);

        // Calculando a saída da camada de saída
        double y3 = calcularSaida(v3, a);
        double y4 = calcularSaida(v4, a);

        // Calculando o erro
        double e3 = calcularErro(y3, dados.get(0)[17]);
        double e4 = calcularErro(y4, dados.get(0)[18]);

        // Calculando o gradiente
        double s3 = calcularGradiente(e3, a, y3);
        double s4 = calcularGradiente(e4, a, y4);

        // Atualizando os neurônios da camada de saída
        vetorW3 = atualizarNeuronios(a, s3, dados.get(0), vetorW3);
        vetorW4 = atualizarNeuronios(a, s4, dados.get(0), vetorW4);
    }

    public static double[] inicializarSinapse(double[] vetorSinapses) {
        Random random = new Random();

        for(int i = 0; i < vetorSinapses.length; i++){
            vetorSinapses[i] = (int) (-1 + 2 * random.nextDouble());
        }
        
        return vetorSinapses;
    }

    public static double treinarV(double[] vetorX, double[] vetorW) {
        double v = 0;
        for(int i = 0;i < vetorX.length; i++) {
            v+= vetorX[i]*vetorW[i];
        }
        return v;
    }

    public static double calcularSaida(double v, double a) {
        double y = 1.0 / (1.0 + Math.exp(-a * v));
        return y;
    }

    public static double calcularErro(double y, double yD) {
        double e = yD-y;
        return e;
    }

    public static double calcularGradiente(double e, double a, double y) {
        double s = e*a*y*(1-y);
        return s;
    }

    public static double[] atualizarNeuronios(double g, double s, double[] entrada, double[] vetorW) {
        for(int i = 0; i< entrada.length; i++){
            vetorW[i] = vetorW[i]+entrada[i]*g*s;
        }
        
        return vetorW;
    }

    public double retropropagar(double a, double y, double s1, double s2, double[] vetorW1, double[] vetorW2) {
        double r = a*y*(1-y)*((s1*vetorW1[1]+s2*vetorW2[1]));
        return r;
    }

    public double erroMedio(double[] vetorErro) {
	double eM = 0;
	for(int i = 0; i < vetorErro.length; i++) {
	    eM += vetorErro[i]*vetorErro[i];
	}
	eM = eM/vetorErro.length;
	
	return eM;
    }

    public double erroMedioEpocas(double[] vetorErroMedio, int epocas) {
    	double eME = 0;

	for(int i = 0; i < vetorErroMedio.length; i++) {
	    eME+= vetorErroMedio[i];
	}
	
	return (1/epocas)*eME;
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
}