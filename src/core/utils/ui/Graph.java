package core.utils.ui;

import info.monitorenter.gui.chart.Chart2D;
import info.monitorenter.gui.chart.ITrace2D;
import info.monitorenter.gui.chart.rangepolicies.RangePolicyFixedViewport;
import info.monitorenter.gui.chart.traces.Trace2DLtd;
import info.monitorenter.util.Range;
import java.awt.Color;
import java.awt.Font;

/**
 *
 * @author gmatoga
 */
public class Graph {

    ITrace2D trace;
    int last;
    Chart2D chart = getChartPreparedForGA();

    public Graph() {

        trace = new Trace2DLtd(10000, "accuracy");
        chart.addTrace(trace);
    }

    public static Chart2D getChartPreparedForGA() {
        Chart2D chart = new Chart2D();
        chart.setToolTipType(Chart2D.ToolTipType.VALUE_SNAP_TO_TRACEPOINTS);
        chart.getAxisY().getAxisTitle().setTitleFont(new Font("Tahoma", Font.PLAIN, 20));
        chart.getAxisX().getAxisTitle().setTitleFont(new Font("Tahoma", Font.PLAIN, 20));
        chart.getAxisY().setRangePolicy(new RangePolicyFixedViewport(new Range(0, 100.001)));
        chart.getAxisY().setMinorTickSpacing(10);
        chart.getAxisX().setMinorTickSpacing(10);
        chart.getAxisX().setMajorTickSpacing(100);
        chart.getAxisY().setPaintGrid(true);
        chart.getAxisX().setPaintGrid(true);
        chart.getAxisX().getAxisTitle().setTitle("Generation");
        chart.getAxisY().getAxisTitle().setTitle("Fitness");
        chart.setGridColor(new Color(225, 225, 240));
        chart.setUseAntialiasing(true);
        return chart;
    }

    public void add(double val) {
        trace.addPoint(last, val);
        last++;
    }
}
