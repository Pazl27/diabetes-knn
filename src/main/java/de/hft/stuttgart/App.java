package de.hft.stuttgart;

import java.util.Scanner;

public class App
{
    public static void main( String[] args ) throws Exception
    {
        final Scanner scanner = new Scanner(App.class.getClassLoader().getResourceAsStream("diabetes.csv"));
        while (scanner.hasNextLine()) {
            System.out.println(scanner.nextLine());
        }

        scanner.close();

    }
}
