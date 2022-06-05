using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class FlowParticle : MonoBehaviour
{
    
    private Vector3 _currentPos;
    private Vector3 _velocity;
    private Vector3 _acceleration;
    private int _minX, _maxX, _minY, _maxY, _minZ, _maxZ;
    private FlowField _flowField;
    private bool _acceptNewPosition = true;
    

    public Vector3 CurrentPos => _currentPos;
    private TrailRenderer _trailRenderer;

    public Vector3 Velocity
    {
        get => _velocity;
        set { if(_acceptNewPosition) _velocity = value; }
    }

    public Vector3 Acceleration
    {
        get => _acceleration;
        set { if(_acceptNewPosition) _acceleration = value; }
    }

    public FlowField Field
    {
        get => _flowField;
        set => _flowField = value;
    }

    public void SetBounds()
    {
        _minX = 0;
        _minY = 0;
        _minZ = 0;

        _maxX = _flowField.Width*(int)(_flowField.Spacing-1);
        _maxY = Field.Height*(int)(_flowField.Spacing-1);
        _maxZ = Field.Depth*(int)(_flowField.Spacing-1);
        _trailRenderer = GetComponent<TrailRenderer>();
    }

    public void SetPosition(Vector3 pos)
    {
        var flagMoved = false;
        var newPos = pos;
        if (pos.x < _minX)
        {
            newPos.x = _maxX;
            flagMoved = true;
        }
        else if (pos.x > _maxX)
        {
            newPos.x = _minX;
            flagMoved = true;
        }

        if (pos.y < _minY)
        {
            newPos.y = _maxY;
            flagMoved = true;
        }
        else if (pos.y > _maxY)
        {
            newPos.y = _minY;
            flagMoved = true;
        }

        if (pos.z < _minZ)
        {
            newPos.z = _maxZ;
            flagMoved = true;
        }
        else if (pos.z > _maxZ)
        {
            newPos.z = _minZ;
            flagMoved = true;
        }

        if (flagMoved)
        {
            _acceptNewPosition = false;
            CreateNewParticle(newPos);
        }

        if (!_acceptNewPosition) return;
        transform.position = newPos;
        _currentPos = newPos;
        
    }

    private void CreateNewParticle(Vector3 pos)
    {
        var newGO = Instantiate(gameObject);
        var p =newGO.GetComponent<FlowParticle>();
        _flowField.SwitchParticle(this, p);
        p.Field = _flowField;
        p.SetBounds();
        p.Acceleration = _acceleration;
        p.Velocity = _velocity;
        p.SetPosition(pos);
        //_trailRenderer.Clear();
    }
}
