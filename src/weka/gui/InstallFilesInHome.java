package weka.gui;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.experiment.RemoteEngine;

/**
 *
 * @author gmatoga
 */
public class InstallFilesInHome {

    /**
     * @param args the command line arguments
     */
    public static void main(String... args) {
        String fromFileName = "GenericObjectEditor_1.props";
        String toFileName = "GenericObjectEditor.props";
        copyFile(fromFileName, toFileName);
        fromFileName = "GenericPropertiesCreator_1.props";
        toFileName = "GenericPropertiesCreator.props";
        copyFile(fromFileName, toFileName);
    }

    private static void copyFile(String fromFileName, String toFileName) {
        try {
            String home = System.getProperty("user.home");
            System.out.println("home = " + home);
            InputStream res = InstallFilesInHome.class.getResourceAsStream(fromFileName);
            System.out.println("available bytes " + res.available());
            File toExtract = new File(home + File.separatorChar + toFileName);
            if (toExtract.length() == res.available())
                return;
            OutputStream out = new FileOutputStream(toExtract);
            System.out.println("writing to: " + toExtract.getAbsolutePath());
            int n;
            while ((n = res.read()) != -1) {
                out.write(n);
            }
            res.close();
            out.close();
            System.out.println(home);
        } catch (IOException ex) {
            Logger.getLogger(InstallFilesInHome.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
