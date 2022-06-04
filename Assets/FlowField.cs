using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FlowField : MonoBehaviour
{
  private Vector3[] _vectors;
  private GameObject[] _arrows;
  [SerializeField] private int width, height, depth;
  [SerializeField] private float spacing;
  [SerializeField] private GameObject arrow;

  private void Start()
  {
    _vectors = new Vector3[width*height*depth];
    _arrows = new GameObject[width*height*depth];
    for (int i = 0; i < width; i++)
    {
      for (int j = 0; j < height; j++)
      {
        for (int k = 0; k < depth; k++)
        {
          var arrowGO = Instantiate(arrow);
          arrowGO.transform.position = new Vector3(i * spacing, j * spacing, k * spacing);
          var id = i + j*width + k*width*height;
          _arrows[id] = arrowGO;
        }
      }
    }
  }

  private void Update()
  {
    for (int i = 0; i < width; i++)
    {
      for (int j = 0; j < height; j++)
      {
        for (int k = 0; k < depth; k++)
        {
          // blockIdx.x + blockIdx.y * gridDim.x  + gridDim.x * gridDim.y * blockIdx.z; 
          var id = i + j*width + k*width*height;
          _vectors[id] = PerlinNoise4D(i, j, k, Time.time);
          _arrows[id].transform.Rotate(_vectors[id]);
        }
      }
    }
  }



    private Vector3 PerlinNoise4D(float x, float y, float z, float w)
    {
      //X coordinate
      float xy = Mathf.PerlinNoise(x, y);
      float xz = Mathf.PerlinNoise(x, z);
      float xw = Mathf.PerlinNoise(x, w);
 
 
      //Ycoordinate
      float yx = Mathf.PerlinNoise(y, x);
      float yz = Mathf.PerlinNoise(y, z);
      float yw = Mathf.PerlinNoise(y, w);
 
 
      //Z coordinate
      float zx = Mathf.PerlinNoise(z, x);
      float zy = Mathf.PerlinNoise(z, y);
      float zw = Mathf.PerlinNoise(z, w);
 
 
      //W coordinate
      float wx = Mathf.PerlinNoise(w, x);
      float wy = Mathf.PerlinNoise(w, y);
      float wz = Mathf.PerlinNoise(w, z);

      var outX = (xy + xz + xw) / 3;
      var outY = (yx + yz + yw) / 3;
      var outZ = (zx + zy + zw) / 3;
      
      return new Vector3(outX,outY,outZ);
      //return (xy + xz + xw + yx + yz + yw + zx + zy + zw + wx + wy + wz)/12;
    }
  
}
