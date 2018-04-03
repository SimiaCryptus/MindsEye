/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.models;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.simiacryptus.mindseye.lang.Tensor;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.hdf5;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.lang.Exception;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.hdf5.*;

/**
 * Class for reading arrays and JSON strings from HDF5 achive files. Originally part of deeplearning4j.
 *
 * @author dave @skymind.io
 */
public class Hdf5Archive {
  private static final Logger log = LoggerFactory.getLogger(Hdf5Archive.class);
  
  static {
    try {
      /* This is necessary for the apply to the BytePointer constructor below. */
      Loader.load(hdf5.class);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
  
  @Nonnull
  private final H5File file;
  @Nonnull
  private final File filename;
  
  /**
   * Instantiates a new Hdf 5 archive.
   *
   * @param filename the archive filename
   */
  public Hdf5Archive(@Nonnull String filename) {
    this(new File(filename));
  }
  
  /**
   * Instantiates a new Hdf 5 archive.
   *
   * @param filename the filename
   */
  public Hdf5Archive(@Nonnull File filename) {
    this.filename = filename;
    try {
      this.file = new H5File(filename.getCanonicalPath(), H5F_ACC_RDONLY());
    } catch (@Nonnull final RuntimeException e) {
      throw e;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  private static void print(@Nonnull Hdf5Archive archive, @Nonnull Logger log) {
    printTree(archive, "", false, log);
  }
  
  private static void printTree(@Nonnull Hdf5Archive hdf5, CharSequence prefix, boolean printData, @Nonnull Logger log, @Nonnull String... path) {
    for (CharSequence datasetName : hdf5.getDataSets(path)) {
      @Nullable Tensor tensor = hdf5.readDataSet(datasetName.toString(), path);
      log.info(String.format("%sDataset %s: %s", prefix, datasetName, Arrays.toString(tensor.getDimensions())));
      if (printData) log.info(String.format("%s%s", prefix, tensor.prettyPrint().replaceAll("\n", "\n" + prefix)));
      tensor.freeRef();
    }
    hdf5.getAttributes(path).forEach((k, v) -> {
      log.info((String.format("%sAttribute: %s => %s", prefix, k, v)));
    });
    for (String t : hdf5.getGroups(path).stream().map(CharSequence::toString).sorted(new Comparator<String>() {
      @Override
      public int compare(@Nonnull String o1, @Nonnull String o2) {
        @Nonnull String prefix = "layer_";
        @Nonnull Pattern digit = Pattern.compile("^\\d+$");
        if (digit.matcher(o1).matches() && digit.matcher(o2).matches())
          return Integer.compare(Integer.parseInt(o1), Integer.parseInt(o2));
        if (o1.startsWith(prefix) && o2.startsWith(prefix))
          return compare(o1.substring(prefix.length()), o2.substring(prefix.length()));
        else return o1.compareTo(o2);
      }
    }).collect(Collectors.toList())) {
      log.info(prefix + t);
      printTree(hdf5, prefix + "\t", printData, log, concat(path, t));
    }
  }
  
  @Nonnull
  private static String[] concat(@Nonnull CharSequence[] s, String t) {
    @Nonnull String[] strings = new String[s.length + 1];
    System.arraycopy(s, 0, strings, 0, s.length);
    strings[s.length] = t;
    return strings;
  }
  
  @Override
  public String toString() {
    return String.format("Hdf5Archive{%s}", file);
  }
  
  @Nonnull
  private Group[] openGroups(@Nonnull CharSequence... groups) {
    @Nonnull Group[] groupArray = new Group[groups.length];
    groupArray[0] = this.file.openGroup(groups[0].toString());
    for (int i = 1; i < groups.length; i++) {
      groupArray[i] = groupArray[i - 1].openGroup(groups[i].toString());
    }
    return groupArray;
  }
  
  private void closeGroups(@Nonnull Group[] groupArray) {
    for (int i = groupArray.length - 1; i >= 0; i--) {
      groupArray[i].deallocate();
    }
  }
  
  /**
   * Read data setBytes as ND4J array from group path.
   *
   * @param datasetName Name of data setBytes
   * @param groups      Array of zero or more ancestor groups from root to parent.
   * @return tensor tensor
   */
  @Nullable
  public Tensor readDataSet(CharSequence datasetName, @Nonnull CharSequence... groups) {
    if (groups.length == 0) {
      return readDataSet(this.file, datasetName);
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    @Nullable Tensor a = readDataSet(groupArray[groupArray.length - 1], datasetName);
    closeGroups(groupArray);
    return a;
  }
  
  /**
   * Read JSON-formatted string attribute from group path.
   *
   * @param attributeName Name of attribute
   * @param groups        Array of zero or more ancestor groups from root to parent.
   * @return string string
   */
  @Nullable
  public CharSequence readAttributeAsJson(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return readAttributeAsJson(this.file.openAttribute(attributeName));
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    @Nullable String s = readAttributeAsJson(groupArray[groups.length - 1].openAttribute(attributeName));
    closeGroups(groupArray);
    return s;
  }
  
  /**
   * Read string attribute from group path.
   *
   * @param attributeName Name of attribute
   * @param groups        Array of zero or more ancestor groups from root to parent.
   * @return string string
   */
  @Nullable
  public CharSequence readAttributeAsString(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return readAttributeAsString(this.file.openAttribute(attributeName));
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    @Nullable String s = readAttributeAsString(groupArray[groupArray.length - 1].openAttribute(attributeName));
    closeGroups(groupArray);
    return s;
  }
  
  /**
   * Check whether group path contains string attribute.
   *
   * @param attributeName Name of attribute
   * @param groups        Array of zero or more ancestor groups from root to parent.
   * @return Boolean indicating whether attribute exists in group path.
   */
  public boolean hasAttribute(String attributeName, @Nonnull String... groups) {
    if (groups.length == 0) {
      return this.file.attrExists(attributeName);
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    boolean b = groupArray[groupArray.length - 1].attrExists(attributeName);
    closeGroups(groupArray);
    return b;
  }
  
  /**
   * Gets attributes.
   *
   * @param groups the groups
   * @return the attributes
   */
  @Nonnull
  public Map<CharSequence, Object> getAttributes(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getAttributes(this.file);
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    Group group = groupArray[groupArray.length - 1];
    @Nonnull Map<CharSequence, Object> attributes = getAttributes(group);
    closeGroups(groupArray);
    return attributes;
  }
  
  /**
   * Gets attributes.
   *
   * @param group the group
   * @return the attributes
   */
  @Nonnull
  public Map<CharSequence, Object> getAttributes(@Nonnull Group group) {
    int numAttrs = group.getNumAttrs();
    @Nonnull TreeMap<CharSequence, Object> attributes = new TreeMap<>();
    for (int i = 0; i < numAttrs; i++) {
      Attribute attribute = group.openAttribute(i);
      CharSequence name = attribute.getName().getString();
      int typeId = attribute.getTypeClass();
      if (typeId == 0) {
        attributes.put(name, getI64(attribute));
      }
      else {
        System.out.println(name + " type = " + typeId);
        attributes.put(name, getString(attribute));
      }
      attribute.deallocate();
    }
    return attributes;
  }
  
  private long getI64(@Nonnull Attribute attribute) {
    return getI64(attribute, attribute.getIntType(), new byte[8]);
  }
  
  @Nonnull
  private CharSequence getString(@Nonnull Attribute attribute) {
    return getString(attribute, attribute.getVarLenType(), new byte[1024]);
  }
  
  private long getI64(@Nonnull Attribute attribute, DataType dataType, @Nonnull byte[] buffer) {
    @Nonnull BytePointer pointer = new BytePointer(buffer);
    attribute.read(dataType, pointer);
    pointer.get(buffer);
    ArrayUtils.reverse(buffer);
    return ByteBuffer.wrap(buffer).asLongBuffer().get();
  }
  
  @Nonnull
  private CharSequence getString(@Nonnull Attribute attribute, DataType dataType, @Nonnull byte[] buffer) {
    @Nonnull BytePointer pointer = new BytePointer(buffer);
    attribute.read(dataType, pointer);
    pointer.get(buffer);
    @Nonnull String str = new String(buffer);
    if (str.indexOf('\0') >= 0) {
      return str.substring(0, str.indexOf('\0'));
    }
    else {
      return str;
    }
  }
  
  /**
   * Get list of data sets from group path.
   *
   * @param groups Array of zero or more ancestor groups from root to parent.
   * @return data sets
   */
  @Nonnull
  public List<CharSequence> getDataSets(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getObjects(this.file, H5O_TYPE_DATASET);
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    @Nonnull List<CharSequence> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_DATASET);
    closeGroups(groupArray);
    return ls;
  }
  
  /**
   * Get list of groups from group path.
   *
   * @param groups Array of zero or more ancestor groups from root to parent.
   * @return groups groups
   */
  @Nonnull
  public List<CharSequence> getGroups(@Nonnull String... groups) {
    if (groups.length == 0) {
      return getObjects(this.file, H5O_TYPE_GROUP);
    }
    @Nonnull Group[] groupArray = openGroups(groups);
    @Nonnull List<CharSequence> ls = getObjects(groupArray[groupArray.length - 1], H5O_TYPE_GROUP);
    closeGroups(groupArray);
    return ls;
  }
  
  /**
   * Read data setBytes as ND4J array from HDF5 group.
   *
   * @param fileGroup   HDF5 file or group
   * @param datasetName Name of data setBytes
   * @return
   */
  @Nullable
  private Tensor readDataSet(@Nonnull Group fileGroup, CharSequence datasetName) {
    DataSet dataset = fileGroup.openDataSet(datasetName.toString());
    DataSpace space = dataset.getSpace();
    int nbDims = space.getSimpleExtentNdims();
    @Nonnull long[] dims = new long[nbDims];
    space.getSimpleExtentDims(dims);
    @Nullable float[] dataBuffer = null;
    @Nullable FloatPointer fp = null;
    int j = 0;
    @Nonnull DataType dataType = new DataType(PredType.NATIVE_FLOAT());
    @Nullable Tensor data = null;
    switch (nbDims) {
      case 4: /* 2D Convolution weights */
        dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2] * dims[3])];
        fp = new FloatPointer(dataBuffer);
        dataset.read(fp, dataType);
        fp.get(dataBuffer);
        data = new Tensor((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
        j = 0;
        for (int i1 = 0; i1 < dims[0]; i1++)
          for (int i2 = 0; i2 < dims[1]; i2++)
            for (int i3 = 0; i3 < dims[2]; i3++)
              for (int i4 = 0; i4 < dims[3]; i4++)
                data.set(i1, i2, i3, i4, (double) dataBuffer[j++]);
        break;
      case 3:
        dataBuffer = new float[(int) (dims[0] * dims[1] * dims[2])];
        fp = new FloatPointer(dataBuffer);
        dataset.read(fp, dataType);
        fp.get(dataBuffer);
        data = new Tensor((int) dims[0], (int) dims[1], (int) dims[2]);
        j = 0;
        for (int i1 = 0; i1 < dims[0]; i1++)
          for (int i2 = 0; i2 < dims[1]; i2++)
            for (int i3 = 0; i3 < dims[2]; i3++)
              data.set(i1, i2, i3, dataBuffer[j++]);
        break;
      case 2: /* Dense and Recurrent weights */
        dataBuffer = new float[(int) (dims[0] * dims[1])];
        fp = new FloatPointer(dataBuffer);
        dataset.read(fp, dataType);
        fp.get(dataBuffer);
        data = new Tensor((int) dims[0], (int) dims[1]);
        j = 0;
        for (int i1 = 0; i1 < dims[0]; i1++)
          for (int i2 = 0; i2 < dims[1]; i2++)
            data.set(i1, i2, dataBuffer[j++]);
        break;
      case 1: /* Bias */
        dataBuffer = new float[(int) dims[0]];
        fp = new FloatPointer(dataBuffer);
        dataset.read(fp, dataType);
        fp.get(dataBuffer);
        data = new Tensor((int) dims[0]);
        j = 0;
        for (int i1 = 0; i1 < dims[0]; i1++)
          data.set(i1, dataBuffer[j++]);
        break;
      default:
        throw new RuntimeException("Cannot import weights apply rank " + nbDims);
    }
    space.deallocate();
    dataset.deallocate();
    return data;
  }
  
  /**
   * Get list of objects apply a given type from a file group.
   *
   * @param fileGroup HDF5 file or group
   * @param objType   Type of object as integer
   * @return
   */
  @Nonnull
  private List<CharSequence> getObjects(@Nonnull Group fileGroup, int objType) {
    @Nonnull List<CharSequence> groups = new ArrayList<CharSequence>();
    for (int i = 0; i < fileGroup.getNumObjs(); i++) {
      BytePointer objPtr = fileGroup.getObjnameByIdx(i);
      if (fileGroup.childObjType(objPtr) == objType) {
        groups.add(fileGroup.getObjnameByIdx(i).getString());
      }
    }
    return groups;
  }
  
  /**
   * Read JSON-formatted string attribute.
   *
   * @param attribute HDF5 attribute to read as JSON formatted string.
   * @return
   */
  @Nullable
  private String readAttributeAsJson(@Nonnull Attribute attribute) {
    VarLenType vl = attribute.getVarLenType();
    int bufferSizeMult = 1;
    @Nullable String s = null;
    /* TODO: find a less hacky way to do this.
     * Reading variable length strings (from attributes) is a giant
     * pain. There does not appear to be any way to determine the
     * length of the string in advance, so we use a hack: choose a
     * buffer size and read the config. If Jackson fails to parse
     * it, then we must not have read the entire config. Increase
     * buffer and repeat.
     */
    while (true) {
      @Nonnull byte[] attrBuffer = new byte[bufferSizeMult * 2000];
      @Nonnull BytePointer attrPointer = new BytePointer(attrBuffer);
      attribute.read(vl, attrPointer);
      attrPointer.get(attrBuffer);
      s = new String(attrBuffer);
      @Nonnull ObjectMapper mapper = new ObjectMapper();
      mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY);
      try {
        mapper.readTree(s);
        break;
      } catch (IOException e) {
      }
      bufferSizeMult++;
      if (bufferSizeMult > 100) {
        throw new RuntimeException("Could not read abnormally long HDF5 attribute");
      }
    }
    return s;
  }
  
  /**
   * Read attribute as string.
   *
   * @param attribute HDF5 attribute to read as string.
   * @return
   */
  @Nullable
  private String readAttributeAsString(@Nonnull Attribute attribute) {
    VarLenType vl = attribute.getVarLenType();
    int bufferSizeMult = 1;
    @Nullable String s = null;
    /* TODO: find a less hacky way to do this.
     * Reading variable length strings (from attributes) is a giant
     * pain. There does not appear to be any way to determine the
     * length of the string in advance, so we use a hack: choose a
     * buffer size and read the config, increase buffer and repeat
     * until the buffer ends apply \u0000
     */
    while (true) {
      @Nonnull byte[] attrBuffer = new byte[bufferSizeMult * 2000];
      @Nonnull BytePointer attrPointer = new BytePointer(attrBuffer);
      attribute.read(vl, attrPointer);
      attrPointer.get(attrBuffer);
      s = new String(attrBuffer);
      
      if (s.endsWith("\u0000")) {
        s = s.replace("\u0000", "");
        break;
      }
      
      bufferSizeMult++;
      if (bufferSizeMult > 100) {
        throw new RuntimeException("Could not read abnormally long HDF5 attribute");
      }
    }
    
    return s;
  }
  
  /**
   * Read string attribute from group path.
   *
   * @param attributeName Name of attribute
   * @param bufferSize    buffer size to read
   * @return string string
   */
  @Nonnull
  public CharSequence readAttributeAsFixedLengthString(String attributeName, int bufferSize) {
    return readAttributeAsFixedLengthString(this.file.openAttribute(attributeName), bufferSize);
  }
  
  /**
   * Read attribute of fixed buffer size as string.
   *
   * @param attribute HDF5 attribute to read as string.
   * @return
   */
  @Nonnull
  private CharSequence readAttributeAsFixedLengthString(@Nonnull Attribute attribute, int bufferSize) {
    VarLenType vl = attribute.getVarLenType();
    @Nonnull byte[] attrBuffer = new byte[bufferSize];
    @Nonnull BytePointer attrPointer = new BytePointer(attrBuffer);
    attribute.read(vl, attrPointer);
    attrPointer.get(attrBuffer);
    @Nonnull String s = new String(attrBuffer);
    return s;
  }
  
  /**
   * Print.
   */
  public void print() {print(log);}
  
  /**
   * Print.
   *
   * @param log the log
   */
  public void print(@Nonnull Logger log) {
    print(this, log);
  }
  
  /**
   * Gets filename.
   *
   * @return the filename
   */
  @Nonnull
  public File getFilename() {
    return filename;
  }
}
