package weka.gui;

import java.awt.*;
import java.io.*;
import javax.swing.*;

public class Console extends JFrame {

    PipedInputStream piOut;
    PipedInputStream piErr;
    PipedOutputStream poOut;
    PipedOutputStream poErr;
    final JTextArea textArea = new JTextArea();

    public Console() throws IOException {
        // Set up System.out
        piOut = new PipedInputStream();
        poOut = new PipedOutputStream(piOut);
        System.setOut(new PrintStream(poOut, true));

        // Set up System.err
        piErr = new PipedInputStream();
        poErr = new PipedOutputStream(piErr);
        System.setErr(new PrintStream(poErr, true));

        // Add a scrolling text area
        textArea.setEditable(false);
        textArea.setRows(20);
        textArea.setColumns(50);
        getContentPane().add(new JScrollPane(textArea), BorderLayout.CENTER);
        pack();
        setVisible(true);

        // Create reader threads
        new ReaderThread(piOut).start();
        new ReaderThread(piErr).start();
    }

    class ReaderThread extends Thread {

        PipedInputStream pi;

        ReaderThread(PipedInputStream pi) {
            this.pi = pi;
        }

        @Override
        public void run() {
            final byte[] buf = new byte[1024 * 20];
            try {
                while (true) {
                    final int len = pi.read(buf);
                    if (len == -1) {
                        break;
                    }
                    SwingUtilities.invokeLater(new Runnable() {

                        public void run() {
                            synchronized (textArea) { // Warning - don't know what I'm doing here
                                textArea.append(new String(buf, 0, len));

                                // Make sure the last line is always visible
                                textArea.setCaretPosition(textArea.getDocument().getLength());

                                // Keep the text area down to a certain character size
                                int idealSize = 1000 * 20;
                                int maxExcess = 2000;
                                int excess = textArea.getDocument().getLength() - idealSize;
                                if (excess >= maxExcess) {
                                    textArea.replaceRange("", 0, excess);
                                }
                            }
                        }
                    });
                }
            } catch (IOException e) {
            }
        }
    }
}
