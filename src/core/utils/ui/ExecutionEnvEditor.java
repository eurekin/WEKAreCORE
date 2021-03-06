
/*
 * ExecutionEnvEditor.java
 *
 */
package core.utils.ui;

import core.ga.ops.ec.ExecutionEnv;
import java.awt.Dimension;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JToggleButton;
import javax.swing.LayoutStyle.ComponentPlacement;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import org.jdesktop.beansbinding.AutoBinding.UpdateStrategy;
import org.jdesktop.beansbinding.BeanProperty;
import org.jdesktop.beansbinding.Binding;
import org.jdesktop.beansbinding.BindingGroup;
import org.jdesktop.beansbinding.Bindings;
import org.jdesktop.beansbinding.ELProperty;

/**
 *
 * @author gmatoga
 */
public class ExecutionEnvEditor extends javax.swing.JPanel {

    ExecutionEnv ec;

    public ExecutionEnv getEc() {
        return ec;
    }

    public void setEc(ExecutionEnv ec) {
        this.ec = ec;
    }

    public ExecutionEnvEditor(ExecutionEnv ec) {
        this.ec = ec;
        initComponents();
        jSpinner1.getModel().addChangeListener(new ChangeListener() {

            public void stateChanged(ChangeEvent e) {
                getEc().setTokenCompetitionWeight((Double) jSpinner1.getModel().getValue());
            }
        });
    }

    /** Creates new form ExecutionEnvEditor */
    public ExecutionEnvEditor() {
        initComponents();

    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        bindingGroup = new BindingGroup();

        jToggleButton1 = new JToggleButton();
        jSpinner1 = new JSpinner();
        jLabel1 = new JLabel();
        jLabel2 = new JLabel();
        jSpinner2 = new JSpinner();
        jSpinner3 = new JSpinner();

        setMinimumSize(new Dimension(214, 97));

        jToggleButton1.setText("Token Competition");

        Binding binding = Bindings.createAutoBinding(UpdateStrategy.READ_WRITE, this, ELProperty.create("${ec.tokenCompetitionEnabled}"), jToggleButton1, BeanProperty.create("selected"));
        bindingGroup.addBinding(binding);

        jSpinner1.setModel(new SpinnerNumberModel(0.0d, 0.0d, 1.0d, 0.1d));

        jLabel1.setText("R Mutation");

        jLabel2.setText("RS Mutation");

        jSpinner2.setModel(new SpinnerNumberModel(0.0d, 0.0d, 1.0d, 0.01d));

        jSpinner3.setModel(new SpinnerNumberModel(0.0d, 0.0d, 1.0d, 0.01d));

        GroupLayout layout = new GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(Alignment.LEADING)
                    .addComponent(jToggleButton1)
                    .addComponent(jLabel1)
                    .addComponent(jLabel2))
                .addPreferredGap(ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(Alignment.LEADING, false)
                    .addComponent(jSpinner3)
                    .addComponent(jSpinner2)
                    .addComponent(jSpinner1, GroupLayout.DEFAULT_SIZE, 67, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(11, 11, 11)
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jToggleButton1)
                    .addComponent(jSpinner1, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addGap(6, 6, 6)
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jLabel1)
                    .addComponent(jSpinner2, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(Alignment.BASELINE)
                    .addComponent(jLabel2)
                    .addComponent(jSpinner3, GroupLayout.PREFERRED_SIZE, GroupLayout.DEFAULT_SIZE, GroupLayout.PREFERRED_SIZE))
                .addContainerGap(GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        bindingGroup.bind();
    }// </editor-fold>//GEN-END:initComponents
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private JLabel jLabel1;
    private JLabel jLabel2;
    private JSpinner jSpinner1;
    private JSpinner jSpinner2;
    private JSpinner jSpinner3;
    private JToggleButton jToggleButton1;
    private BindingGroup bindingGroup;
    // End of variables declaration//GEN-END:variables
}
