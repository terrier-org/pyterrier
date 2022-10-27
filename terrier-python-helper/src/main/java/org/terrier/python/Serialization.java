package org.terrier.python;
import java.io.IOException;
import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;
/** 
 * implementations adapted from https://stackoverflow.com/a/2836659 */
public class Serialization {

    public static byte[] serialize(Serializable obj) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream out = null;
        byte[] rtr = new byte[0];
        try {
            out = new ObjectOutputStream(bos);   
            out.writeObject(obj);
            out.flush();
            rtr = bos.toByteArray();
        } finally {
            try {
                bos.close();
            } catch (IOException ex) {
                // ignore close exception
            }
        }
        return rtr;
    }

    @SuppressWarnings("unchecked")
    public static <K extends Serializable> K deserialize(byte[] input, Class<K> clz) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bis = new ByteArrayInputStream(input);
        ObjectInputStream in = null;
        Object rtr = null;
        try {
            in = new ObjectInputStream(bis);
            rtr = in.readObject(); 
        } finally {
            try {
                if (in != null) {
                in.close();
                }
            } catch (IOException ex) {
                // ignore close exception
            }
        }
        return (K)rtr;
    }


}
