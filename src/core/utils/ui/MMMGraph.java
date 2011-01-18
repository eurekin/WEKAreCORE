package core.utils.ui;

import core.stat.SimpleStatistics;
import info.monitorenter.gui.chart.Chart2D;
import info.monitorenter.gui.chart.ITrace2D;
import info.monitorenter.gui.chart.traces.Trace2DLtd;
import java.awt.BasicStroke;
import java.awt.Color;

/**
 *
 * @author gmatoga
 */
public class MMMGraph {

    ITrace2D min;
    int last;
    private final Trace2DLtd max;
    private final Trace2DLtd mean;
    Chart2D chart = Graph.getChartPreparedForGA();
    private LinearBarGraphPanel p;

    public Chart2D getChart() {
        return chart;
    }

    public MMMGraph(String title) {

        min = new Trace2DLtd(10000, "min");
        max = new Trace2DLtd(10000, "max");
        mean = new Trace2DLtd(10000, "mean");
        max.setColor(Color.red);
        max.setStroke(new BasicStroke(2.1f, BasicStroke.CAP_BUTT,BasicStroke.JOIN_BEVEL));
        min.setColor(Color.lightGray);
        chart.addTrace(min);
        chart.addTrace(max);
        chart.addTrace(mean);
    }

    public void add(SimpleStatistics stats) {
        min.addPoint(last, stats.getMin() * 100);
        max.addPoint(last, stats.getMax() * 100);
        mean.addPoint(last, stats.getMean() * 100);
        setVal(stats.getMax());
        last++;
    }

    public void setPanel(LinearBarGraphPanel linearBarGraphPanel) {
        this.p = linearBarGraphPanel;
    }

    private void setVal(double max) {
        if (p != null) {
            p.setValue(max * 100);
        }
    }
}
