### Json Serialization
Code from [LayerTestBase.java:74](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L74) executed in 0.01 seconds: 
```java
  
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.DropoutNoiseLayer",
      "id": "bdd6bbba-380b-47fe-a761-c24100016356",
      "isFrozen": false,
      "name": "DropoutNoiseLayer/bdd6bbba-380b-47fe-a761-c24100016356",
      "inputs": [
        "0d038d42-bff9-4bae-be4e-b67ec135a7df"
      ],
      "nodes": {
        "0b30f42a-1d3f-4275-bf89-5b53f5fccdf8": "bdd6bbba-380b-47fe-a761-c24100016358",
        "fc23fd90-a497-483c-ac4b-4c3f5ab9e9b9": "bdd6bbba-380b-47fe-a761-c24100016357"
      },
      "layers": {
        "bdd6bbba-380b-47fe-a761-c24100016358": {
          "class": "com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer",
          "id": "bdd6bbba-380b-47fe-a761-c24100016358",
          "isFrozen": false,
          "name": "mask",
          "value": 0.5
        },
        "bdd6bbba-380b-47fe-a761-c24100016357": {
          "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ProductInputsLayer",
          "id": "bdd6bbba-380b-47fe-a761-c24100016357",
          "isFrozen": false,
          "name": "ProductInputsLayer/bdd6bbba-380b-47fe-a761-c24100016357"
        }
      },
      "links": {
        "0b30f42a-1d3f-4275-bf89-5b53f5fccdf8": [
          "0d038d42-bff9-4bae-be4e-b67ec135a7df"
        ],
        "fc23fd90-a497-483c-ac4b-4c3f5ab9e9b9": [
          "0b30f42a-1d3f-4275-bf89-5b53f5fccdf8",
          "0d038d42-bff9-4bae-be4e-b67ec135a7df"
        ]
      },
      "labels": {},
      "head": "fc23fd90-a497-483c-ac4b-4c3f5ab9e9b9"
    }
```



### Network Diagram
Code from [LayerTestBase.java:85](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L85) executed in 3.73 seconds: 
```java
    log.h3("Network Diagram");
    log.code(()->{
      return Graphviz.fromGraph(toGraph((DAGNetwork) layer))
        .height(400).width(600).render(Format.PNG).toImage();
    });
```

Returns: 

![Result](etc/test.1.png)



### Differential Validation
Code from [LayerTestBase.java:98](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L98) executed in 0.20 seconds: 
```java
  
```
Logging: 
```
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.6374e-06 +- 2.4136e-05 [0.0000e+00 - 1.6594e-04] (324#)
    relativeTol: 1.4730e-04 +- 4.9276e-05 [1.6928e-05 - 1.6593e-04] (8#)
    
```

### Performance
Code from [LayerTestBase.java:103](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L103) executed in 14.22 seconds: 
```java
  
```
Logging: 
```
    Forward performance: 1421.6459 +- 67.8565 [1165.1968 - 1804.2181]
    
```

### Reference Implementation
