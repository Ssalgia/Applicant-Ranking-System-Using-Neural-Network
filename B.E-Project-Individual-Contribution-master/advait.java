import java.util.*;
import java.io.*;
class advait
{
public static void main(String[] args) 
{
	System.out.println("Input the sentence");
        Scanner s = new Scanner(System.in);
        String input = s.nextLine();

        System.out.println("Input the word that has to be searched");
        String word = s.nextLine();

        String str = "";
        int occurance = 0;
        for(char c : input.toCharArray()) 
	{
         str += c;
         if(str.length() == word.length()) {
             if(str.equals(word)) {
                 occurance ++;
             }
	str = str.substring(1);
         }
     }

     if(occurance > 0)
         System.out.println("the string is present and the count of the given string is : " + occurance);
     else 
         System.out.println("The string is not present");
}
}
