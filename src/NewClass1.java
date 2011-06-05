
import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;


/**
 *
 * @author gmatoga
 */
public class NewClass1 {
private final String name = "SimpleBean";
private int size;

public String getName()
{
    return this.name;
}

public int getSize()
{
    return this.size;
}

public NewClass1 setSize( int size )
{
    this.size = size;
    return this;
}

public static void main( String[] args )
        throws IntrospectionException
{
    BeanInfo info = Introspector.getBeanInfo( NewClass1.class );
    for ( PropertyDescriptor pd : info.getPropertyDescriptors() )
        System.out.println( pd.getName() );
}
}
