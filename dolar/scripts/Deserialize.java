
import java.io.*;
import java.util.Scanner;
public class Deserialize {


    public static void main(String[] args) {
        try {
            FileInputStream fis = new FileInputStream("tryd_replay_20200601.0.0.dat");
            Scanner scnr = new Scanner(fis);
            //DataInputStream  dados = new DataInputStream(fis);
            //Object object = ois.readObject();
  

          

            while(scnr.hasNextLine()){
                String line = scnr.nextLine();
                System.out.println(line);
             }

        } catch (Exception e) {
            System.out.println("Erro ao carregar");
            e.printStackTrace();
        }
    

    }
}


