/*
 * Copyright (c) 2017 by Andrew Charneski.
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

package com.simiacryptus.util.lang;

import org.apache.commons.io.IOUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Code util.
 */
public class CodeUtil {
  /**
   * The constant projectRoot.
   */
  public static File projectRoot = new File(System.getProperty("codeRoot", ".."));
  private static final List<File> codeRoots = loadCodeRoots();
  
  private static List<File> loadCodeRoots() {
    List<String> folders = Arrays.asList(
      "src/main/java", "src/test/java", "src/main/scala", "src/test/scala"
    );
    List<File> codeLocation = folders.stream().map(name -> new File(projectRoot, name))
      .filter(file -> file.exists() && file.isDirectory()).collect(Collectors.toList());
    if (codeLocation.isEmpty()) {
      codeLocation = Arrays.stream(projectRoot.listFiles()).filter(x -> x.isDirectory()).flatMap(childRoot ->
        folders.stream().map(name -> new File(childRoot, name)).filter(file -> file.exists() && file.isDirectory()))
        .collect(Collectors.toList());
    }
    return codeLocation;
  }
  
  
  /**
   * Find file file.
   *
   * @param callingFrame the calling frame
   * @return the file
   */
  public static File findFile(StackTraceElement callingFrame) {
    String[] packagePath = callingFrame.getClassName().split("\\.");
    String path = Arrays.stream(packagePath).limit(packagePath.length - 1).collect(Collectors.joining(File.separator)) + File.separator + callingFrame.getFileName();
    return findFile(path);
  }
  
  /**
   * Gets indent.
   *
   * @param txt the txt
   * @return the indent
   */
  public static String getIndent(String txt) {
    Matcher matcher = Pattern.compile("^\\s+").matcher(txt);
    return matcher.find() ? matcher.group(0) : "";
  }
  
  /**
   * Find file file.
   *
   * @param path the path
   * @return the file
   */
  public static File findFile(String path) {
    for (File root : codeRoots) {
      File file = new File(root, path);
      if (file.exists()) return file;
    }
    throw new RuntimeException(String.format("Not Found: %s; Project Root = %s", path, projectRoot.getAbsolutePath()));
  }
  
  /**
   * Gets javadoc.
   *
   * @param clazz the clazz
   * @return the javadoc
   */
  public static String getJavadoc(Class<?> clazz) {
    try {
      File source = findFile(clazz);
      if (null == source) return clazz.getName() + " not found";
      List<String> lines = IOUtils.readLines(new FileInputStream(source), Charset.forName("UTF-8"));
      int classDeclarationLine = IntStream.range(0, lines.size())
        .filter(i -> lines.get(i).contains("class " + clazz.getSimpleName())).findFirst().getAsInt();
      int firstLine = IntStream.rangeClosed(1, classDeclarationLine).map(i -> classDeclarationLine - i)
        .filter(i -> !lines.get(i).matches("\\s*[/\\*]{1,4}.*")).findFirst().orElse(-1) + 1;
      String javadoc = lines.subList(firstLine, classDeclarationLine).stream()
        .map(s -> s.replaceFirst("^[ \t]*[/\\*]+", "").trim())
        .filter(x -> !x.isEmpty()).reduce((a, b) -> a + "\n" + b).orElse("");
      return javadoc;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Find file file.
   *
   * @param clazz the clazz
   * @return the file
   */
  public static File findFile(Class<?> clazz) {
    String path = clazz.getName().replaceAll("\\.", "/").replaceAll("\\$.*", "");
    return findFile(path + ".java");
  }
  
  /**
   * Gets inner text.
   *
   * @param callingFrame the calling frame
   * @return the inner text
   * @throws IOException the io exception
   */
  public static String getInnerText(StackTraceElement callingFrame) throws IOException {
    try {
      File file = findFile(callingFrame);
      assert (null != file);
      int start = callingFrame.getLineNumber() - 1;
      List<String> allLines = Files.readAllLines(file.toPath());
      String txt = allLines.get(start);
      String indent = getIndent(txt);
      ArrayList<String> lines = new ArrayList<>();
      for (int i = start + 1; i < allLines.size() && (getIndent(allLines.get(i)).length() > indent.length() || allLines.get(i).trim().isEmpty()); i++) {
        String line = allLines.get(i);
        lines.add(line.substring(Math.min(indent.length(), line.length())));
      }
      return lines.stream().collect(Collectors.joining("\n"));
    } catch (Throwable e) {
      return "";
    }
  }
}
