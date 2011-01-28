package core.ui;


import core.utils.ui.LinearBarGraphPanel;
import core.utils.ui.MMMGraph;
import core.utils.ui.MultiPanelFrame;

public class evoGUI {

    public evoGUI(MMMGraph graphR, MMMGraph graphRS) {
        MultiPanelFrame frame = new MultiPanelFrame();
        frame.getCenterPanel().add(graphR.getChart());
        frame.getCenterPanel().add(graphRS.getChart());
        LinearBarGraphPanel linearBarGraphPanel = new LinearBarGraphPanel();
        frame.setLocationByPlatform(true);
        graphRS.setPanel(linearBarGraphPanel);
        frame.getBottomPanel().add(linearBarGraphPanel);
        frame.setVisible(true);
    }
}
