#!/usr/bin/env python3
"""
Cocos Creator Pack Inspector
===========================

A comprehensive single-file viewer for Cocos Creator 2.x import pack files.

This inspector analyzes packed JSON formats from Cocos Creator projects and provides:
- Scene graph visualization with node hierarchy
- Component property decoding (Labels, Sprites, Buttons, etc.)
- Asset reference tracking and mapping
- Transform data extraction (position, rotation, scale, anchor)
- Value type deserialization (Vec2, Vec3, Color, etc.)
- Property index to name resolution using CLASS_INDEX mappings

Format Specification - How the Inspector Maps Indices → Properties:
================================================================

Pack File Structure:
    data[0] = format_version
    data[1] = uuids/dependencies array
    data[2] = property_names array  # Global property name lookup
    data[3] = class_definitions     # Class templates with property indices
    data[4] = templates            # Template instances mapping to classes
    data[5+] = actual node/component data blocks

Template Index Resolution:
    1. Component has template_index (e.g., 5)
    2. templates[5] = [class_def_index, ...extra_data]
    3. class_definitions[class_def_index] = ["cc.Label", [prop_idx1, prop_idx2, ...]]
    4. property_names[prop_idx1] = "_string", property_names[prop_idx2] = "_fontSize"
    5. Component data: [5, "Hello", 24] → template_index=5, _string="Hello", _fontSize=24

Component Data Layout:
    [template_index, prop_value1, prop_value2, ...]
    
    Where properties map to:
    - template_index → CLASS_INDEX[template_index]["class"] and ["props"]
    - prop_valueN → property_names[class_definitions[...][N]]

Node Data Layout (packed rows):
    [0] = template_index (maps to node class)
    [1] = name (string)
    [2] = parent_index (int)
    [3] = components array [[comp_template, prop1, prop2, ...], ...]
    [4] = additional data
    [5] = size data [flag, width, height] or value-type array
    [6] = transform data [px,py,pz, rx,ry,rz, sx,sy,sz, ...]
    [7+] = additional node-specific data

Value Type Arrays:
    [type_id, ...components] where type_id ∈ {0..7}
    - [0, x, y] → Vec2(x, y)
    - [1, x, y, z] → Vec3(x, y, z)
    - [4, packed_argb] → Color(r, g, b, a)
    - [5, width, height] → Size(width, height)
    - [6, x, y, w, h] → Rect(x, y, w, h)

Asset References:
    - Negative indices: Often template references
    - Large positive (>10000): Typically asset IDs
    - Small positive: Node indices or local references

Usage Examples:
    # Basic inspection
    python test.py bundle_config.json
    
    # Show all assets (not just referenced)
    python test.py bundle_config.json --assets
    
    # Test component decoder engine
    python test.py --test
"""

from __future__ import annotations
import contextlib, argparse
import json, sys
from pathlib import Path
from typing  import Any, Dict, List, Tuple, Set, Optional, Type
from dataclasses import dataclass, field
from enum import IntEnum

# Heuristic threshold for detecting numeric asset references.
# Cocos Creator packs often use large integer values for asset IDs.
# Values greater than this constant are assumed to reference assets.
# Adjust to tune asset detection for different pack formats.
ASSET_REF_THRESHOLD = 10000

# ────────────────────────── packed-row column indices ─────────────────────────
IDX_OVR   = 3          # components / overrides list
IDX_SIZE  = 5          # [flag, width, height]
IDX_TRS   = 6          # [px,py,pz, rx,ry,rz, sx,sy,sz, …]

# ────────────── Engine Value Types and Deserializer ──────────────

class ValueTypeID(IntEnum):
    Vec2  = 0
    Vec3  = 1
    Vec4  = 2
    Quat  = 3
    Color = 4
    Size  = 5
    Rect  = 6
    Mat4  = 7

@dataclass
class Vec2:
    x: float
    y: float
    def __str__(self): return f"({self.x},{self.y})"

@dataclass
class Vec3:
    x: float
    y: float
    z: float
    def __str__(self): return f"({self.x},{self.y},{self.z})"

@dataclass
class Vec4:
    x: float
    y: float
    z: float
    w: float
    def __str__(self): return f"({self.x},{self.y},{self.z},{self.w})"

@dataclass
class Quat:
    x: float
    y: float
    z: float
    w: float
    def __str__(self): return f"quat({self.x},{self.y},{self.z},{self.w})"

@dataclass
class Rect:
    x: float
    y: float
    width: float
    height: float
    def __str__(self): return f"rect({self.x},{self.y},{self.width},{self.height})"

@dataclass
class Mat4:
    m00: float; m01: float; m02: float; m03: float
    m10: float; m11: float; m12: float; m13: float
    m20: float; m21: float; m22: float; m23: float
    m30: float; m31: float; m32: float; m33: float
    def __str__(self): return f"mat4([{self.m00},{self.m01},{self.m02},{self.m03}|{self.m10},{self.m11},{self.m12},{self.m13}|{self.m20},{self.m21},{self.m22},{self.m23}|{self.m30},{self.m31},{self.m32},{self.m33}])"

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int
    def __str__(self): return f"rgba({self.r},{self.g},{self.b},{self.a})"

@dataclass
class Size:
    width: float
    height: float
    def __str__(self): return f"({self.width}×{self.height})"

class ValueTypeDeserializer:
    """Central value-type array decoder for Cocos Creator format.
    
    Handles deserialization of packed value-type arrays used throughout
    Cocos Creator pack formats for vectors, colors, rectangles, etc.
    
    Format: [type_id, ...components] where type_id maps to ValueTypeID enum.
    """
    @staticmethod
    def is_value_array(arr: Any) -> bool:
        """Check if an array follows the value-type format [type_id, ...data].
        
        Args:
            arr: Potential value-type array to check
            
        Returns:
            True if arr is a valid value-type array format
        """
        return (isinstance(arr, list) and 
                len(arr) > 0 and 
                isinstance(arr[0], int) and 
                0 <= arr[0] <= 7)

    @staticmethod
    def decode(arr: List[Any]) -> Any:
        """Decode a value-type array into its corresponding dataclass.
        
        Args:
            arr: Value-type array in format [type_id, ...components]
            
        Returns:
            Decoded value object (Vec2, Vec3, Color, etc.) or original array if unknown type
            
        Examples:
            decode([0, 10.5, 20.0]) → Vec2(10.5, 20.0)
            decode([4, 0xFF8080FF]) → Color(128, 128, 255, 255)
            decode([5, 100, 50]) → Size(100.0, 50.0)
        """
        vt_id, comps = arr[0], arr[1:]
        
        if vt_id == ValueTypeID.Vec2:
            return Vec2(comps[0], comps[1])
        
        if vt_id == ValueTypeID.Vec3:
            return Vec3(comps[0], comps[1], comps[2])
        
        if vt_id == ValueTypeID.Vec4:
            return Vec4(comps[0], comps[1], comps[2], comps[3])
        
        if vt_id == ValueTypeID.Quat:
            return Quat(comps[0], comps[1], comps[2], comps[3])
        
        if vt_id == ValueTypeID.Color:
            if len(comps) == 1 and isinstance(comps[0], int):
                # Packed color format (ARGB)
                packed = comps[0]
                a = (packed >> 24) & 0xFF
                r = (packed >> 16) & 0xFF
                g = (packed >> 8)  & 0xFF
                b = (packed >> 0)  & 0xFF
                return Color(r, g, b, a)
            elif len(comps) == 4:
                # RGBA list format
                return Color(int(comps[0]), int(comps[1]), int(comps[2]), int(comps[3]))
            elif len(comps) == 3:
                # RGB list format (assume full alpha)
                return Color(int(comps[0]), int(comps[1]), int(comps[2]), 255)
            else:
                # Fallback to white
                return Color(255, 255, 255, 255)
        
        if vt_id == ValueTypeID.Size:
            return Size(comps[0], comps[1])
        
        if vt_id == ValueTypeID.Rect:
            return Rect(comps[0], comps[1], comps[2], comps[3])
        
        if vt_id == ValueTypeID.Mat4:
            if len(comps) >= 16:
                return Mat4(
                    comps[0], comps[1], comps[2], comps[3],
                    comps[4], comps[5], comps[6], comps[7],
                    comps[8], comps[9], comps[10], comps[11],
                    comps[12], comps[13], comps[14], comps[15]
                )
            else:
                # Fallback for incomplete matrix data
                padded = comps + [0.0] * (16 - len(comps))
                return Mat4(
                    padded[0], padded[1], padded[2], padded[3],
                    padded[4], padded[5], padded[6], padded[7],
                    padded[8], padded[9], padded[10], padded[11],
                    padded[12], padded[13], padded[14], padded[15]
                )
        
        return arr
    
    @staticmethod
    def decode_ref(idx: int) -> Tuple[str, int]:
        """Decode template/asset reference indices to (kind, idx) tuple.
        
        Args:
            idx: The index value to decode
            
        Returns:
            Tuple of (kind, idx) where kind is one of:
            - 'template': References a template in the templates array
            - 'asset': References an asset in the assets table
            - 'node': References a node by index
            - 'unknown': Unknown reference type
        """
        if not isinstance(idx, int):
            return ('unknown', idx)
        
        # Cocos Creator reference patterns based on common pack formats:
        # - Negative indices often reference templates or special objects
        # - Large positive indices often reference assets
        # - Small positive indices often reference nodes or local objects
        
        if idx < 0:
            # Negative indices typically reference templates
            return ('template', abs(idx))
        elif idx > ASSET_REF_THRESHOLD:  # Threshold for asset references (configurable)
            # Large positive indices typically reference assets
            return ('asset', idx)
        elif idx > 0:
            # Small positive indices typically reference nodes
            return ('node', idx)
        else:
            # idx == 0, could be null reference or root node
            return ('node', idx)

# ──────────────────────── CLASS INDEX MAPPING REFERENCE ───────────────────────

# Comprehensive class index mapping derived from pack format analysis
CLASS_INDEX_REFERENCE = {
    # Core Cocos Creator Classes
    "cc.Node": {
        "common_props": ["_name", "_active", "_parent", "_components", "_prefab", "_contentSize", "_trs", "_children", "_anchorPoint", "_color"],
        "full_props": ["_name", "_opacity", "_objFlags", "_active", "_id", "_components", "_contentSize", "_parent", "_prefab", "_trs", "_children", "_anchorPoint", "_color"]
    },
    "cc.Label": {
        "props": ["_N$verticalAlign", "_N$horizontalAlign", "_string", "_fontSize", "_isSystemFontUsed", "_N$cacheMode", "_lineHeight", "_styleFlags", "_N$overflow", "_enableWrapText", "node", "_materials", "_N$file"]
    },
    "cc.Sprite": {
        "props": ["_sizeMode", "_type", "_isTrimmedMode", "_enabled", "_dstBlendFactor", "_fillRange", "node", "_materials", "_spriteFrame"]
    },
    "cc.Button": {
        "props": ["zoomScale", "_N$transition", "_N$enableAutoGrayEffect", "node", "clickEvents", "_N$pressedColor", "_N$disabledColor", "_N$target", "_N$normalColor", "_N$normalSprite", "_N$pressedSprite", "_N$hoverSprite", "_N$disabledSprite"]
    },
    "cc.Widget": {
        "props": ["_alignFlags", "_originalWidth", "_left", "_right", "_bottom", "_top", "_originalHeight", "alignMode", "_enabled", "node", "_target"]
    },
    "cc.Animation": {
        "props": ["playOnLoad", "node", "_clips", "_defaultClip"]
    },
    "cc.AnimationClip": {
        "props": ["_name", "_duration", "sample", "wrapMode", "speed", "curveData"]
    },
    "cc.ParticleSystem": {
        "props": ["_dstBlendFactor", "_custom", "totalParticles", "emissionRate", "life", "angle", "angleVar", "speed", "tangentialAccel", "lifeVar", "startSize", "speedVar", "endSize", "_positionType", "endRadius", "startSizeVar", "endSizeVar", "endSpinVar", "emitterMode", "endRadiusVar", "startRadius", "duration", "radialAccelVar", "node", "_materials", "_startColor", "_startColorVar", "_endColor", "_endColorVar", "posVar", "_file", "_spriteFrame", "gravity"]
    },
    "cc.ProgressBar": {
        "props": ["_N$mode", "_N$progress", "_N$totalLength", "node", "_N$barSprite"]
    },
    "cc.Slider": {
        "props": ["_N$progress", "node", "slideEvents", "_N$handle"]
    },
    "cc.ScrollView": {
        "props": ["horizontal", "brake", "bounceDuration", "_N$horizontalScrollBar", "_N$verticalScrollBar", "node", "_N$content"]
    },
    "cc.Layout": {
        "props": ["_resize", "_N$layoutType", "_N$paddingLeft", "_N$spacingX", "_N$spacingY", "_N$paddingRight", "_enabled", "node", "_layoutSize"]
    },
    "cc.Mask": {
        "props": ["_N$alphaThreshold", "_type", "node", "_materials", "_spriteFrame"]
    },
    "cc.Canvas": {
        "props": ["_fitWidth", "node", "_designResolution"]
    },
    "cc.Camera": {
        "props": ["_clearFlags", "_depth", "node"]
    },
    "cc.Scene": {
        "props": ["_name", "_active", "autoReleaseAssets", "_children", "_anchorPoint", "_trs"]
    }
}

@dataclass
class ClassTemplateInfo:
    """Enhanced class template information with property mapping.
    
    Represents a resolved template mapping from pack data, connecting
    template indices to actual class names and property lists.
    
    Attributes:
        tpl_index: Template index in the templates array
        class_name: Cocos Creator class name (e.g., "cc.Label", "cc.Sprite")
        properties: Ordered list of property names for this template
        extra_data: Additional template data beyond the basic class definition
    """
    tpl_index: int
    class_name: str
    properties: List[str]
    extra_data: List[int]
    
    def get_property_at_index(self, index: int) -> Optional[str]:
        """Get property name at specific index, if available.
        
        Args:
            index: Property index to look up
            
        Returns:
            Property name if index is valid, None otherwise
            
        Example:
            template.get_property_at_index(0) → "_string" (for cc.Label)
        """
        if 0 <= index < len(self.properties):
            return self.properties[index]
        return None

@dataclass 
class PackFormatInfo:
    """Complete pack format metadata and index resolution system.
    
    This class handles the complex mapping from pack file structure to
    usable class and property information. It builds the bridge between
    numeric indices in packed data and human-readable property names.
    
    Attributes:
        format_version: Pack format version number
        uuids_count: Number of UUID/dependency entries
        property_names: Global property name lookup table
        class_definitions: Raw class definition data from pack
        templates: Raw template data from pack
        class_templates: Resolved template index → ClassTemplateInfo mapping
        
    The key insight: This class resolves the indirection:
        template_index → class_definition → property_indices → property_names
    """
    format_version: int
    uuids_count: int
    property_names: List[str]
    class_definitions: List[Any]
    templates: List[Any]
    class_templates: Dict[int, ClassTemplateInfo] = field(default_factory=dict)
    
    def build_class_index(self) -> None:
        """Build comprehensive class index mapping from pack data.
        
        This is the core method that resolves the template system:
        1. Iterates through templates array
        2. Maps each template to its class definition
        3. Resolves property indices to actual property names
        4. Populates class_templates dictionary
        
        The result enables direct lookup: template_index → class info + properties
        """
        prop_ref = {i: name for i, name in enumerate(self.property_names)}
        
        for tpl_idx, template in enumerate(self.templates):
            if not isinstance(template, list) or len(template) < 2:
                continue
            
            class_def_idx = template[0]
            if not (0 <= class_def_idx < len(self.class_definitions)):
                continue
                
            class_def = self.class_definitions[class_def_idx]
            if not isinstance(class_def, list) or len(class_def) < 2:
                continue
                
            class_name = class_def[0]
            prop_indices = class_def[1:] if len(class_def) > 1 else []
            
            # Map property indices to names
            properties = []
            for prop_idx in prop_indices:
                if isinstance(prop_idx, int):
                    if prop_idx < 0:  # Negative indices reference property_names
                        ref_idx = (-prop_idx) - 1
                        if 0 <= ref_idx < len(self.property_names):
                            properties.append(self.property_names[ref_idx])
                    else:  # Positive indices might be direct references
                        properties.append(f"prop_{prop_idx}")
                else:
                    # Handle non-integer indices safely
                    properties.append(f"unknown_{prop_idx}")
            
            # Use reference data if available
            if class_name in CLASS_INDEX_REFERENCE:
                ref_props = CLASS_INDEX_REFERENCE[class_name].get("props", [])
                if not ref_props:
                    ref_props = CLASS_INDEX_REFERENCE[class_name].get("common_props", [])
                if ref_props and len(ref_props) >= len(properties):
                    properties = ref_props[:len(properties)]
            
            extra_data = template[2:] if len(template) > 2 else []
            
            self.class_templates[tpl_idx] = ClassTemplateInfo(
                tpl_index=tpl_idx,
                class_name=class_name,
                properties=properties,
                extra_data=extra_data
            )
    
    @classmethod
    def from_pack_data(cls, data: List[Any]) -> "PackFormatInfo":
        """Create PackFormatInfo from raw pack data."""
        if len(data) < 5:
            raise ValueError("Invalid pack data structure")
        
        format_info = cls(
            format_version=data[0],
            uuids_count=len(data[1]) if isinstance(data[1], list) else 0,
            property_names=data[2] if isinstance(data[2], list) else [],
            class_definitions=data[3] if isinstance(data[3], list) else [],
            templates=data[4] if isinstance(data[4], list) else []
        )
        
        format_info.build_class_index()
        return format_info

# ─────────────────────────────── util formatters ──────────────────────────────
def _num(x: Any) -> str:
    try:
        return ("{:.6g}".format(float(x))).rstrip("0").rstrip(".")
    except Exception:
        return "-"

def _vec(tab: List[Any], base: int) -> Tuple[str, str, str]:
    try:
        return tuple(_num(tab[base + k]) for k in range(3))
    except Exception:
        return ("-", "-", "-")

# ───────────────────────────── ComponentDecoder Engine ───────────────────────────────

from abc import ABC, abstractmethod

@dataclass(frozen=True)
class AssetReference:
    """Represents a reference to an asset in the pack format."""
    asset_id: int
    asset_type: str = "unknown"
    
    def __str__(self) -> str:
        return f"Asset({self.asset_id}, {self.asset_type})"

@dataclass
class DecodedComponent:
    """Base class for all decoded components."""
    component_type: str
    template_index: int
    properties: Dict[str, Any] = field(default_factory=dict)
    asset_refs: Set[AssetReference] = field(default_factory=set)
    
    def add_asset_ref(self, asset_id: int, asset_type: str = "unknown"):
        """Add an asset reference to this component."""
        self.asset_refs.add(AssetReference(asset_id, asset_type))
    
    def __str__(self) -> str:
        """String representation showing component type and key properties."""
        props_str = ""
        if self.properties:
            # Show up to 3 most relevant properties
            key_props = []
            for key, value in list(self.properties.items())[:3]:
                if value is not None and value != "" and value != []:
                    if isinstance(value, str):
                        key_props.append(f"{key}='{value}'")
                    else:
                        key_props.append(f"{key}={value}")
            props_str = f"({', '.join(key_props)})" if key_props else ""
        
        asset_str = f" [assets: {len(self.asset_refs)}]" if self.asset_refs else ""
        return f"{self.component_type}{props_str}{asset_str}"

@dataclass 
class DecodeHelper:
    """Helper object containing context and utilities for decoding."""
    template_info: Optional[ClassTemplateInfo] = None
    asset_registry: Dict[int, Any] = field(default_factory=dict)
    property_names: List[str] = field(default_factory=list)
    value_type_decoder: Optional[Any] = None
    
    def is_asset_reference(self, value: Any) -> bool:
        """Check if a value looks like an asset reference."""
        return isinstance(value, int) and value in self.asset_registry
    
    def get_asset_type(self, asset_id: int) -> str:
        """Get the type of an asset by its ID."""
        if asset_id in self.asset_registry:
            asset_info = self.asset_registry[asset_id]
            if hasattr(asset_info, 'cls'):
                return asset_info.cls
            elif isinstance(asset_info, dict) and 'class' in asset_info:
                return asset_info['class']
        return "unknown"
    
    def decode_value_type(self, value: Any) -> Any:
        """Decode value types using the provided decoder."""
        if self.value_type_decoder and isinstance(value, list) and ValueTypeDeserializer.is_value_array(value):
            return ValueTypeDeserializer.decode(value)
        return value

class ComponentDecoder(ABC):
    """Abstract base class for component decoders."""
    
    @abstractmethod
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> DecodedComponent:
        """Decode a raw component row into a structured component object."""
        pass
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return the component type this decoder handles."""
        pass

# ───────────────────────────── Specific Decoders ───────────────────────────────

@dataclass
class AnimationClip:
    """Represents an animation clip with keyframes."""
    name: str
    duration: float
    curves: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    wrap_mode: str = "normal"
    
    def __str__(self) -> str:
        return f"AnimClip('{self.name}', {self.duration}s, {len(self.curves)} curves)"

@dataclass
class AnimationComponent(DecodedComponent):
    """Decoded cc.Animation component."""
    play_on_load: bool = False
    clips: List[AnimationClip] = field(default_factory=list)
    default_clip: Optional[str] = None
    
    def __post_init__(self):
        self.component_type = "cc.Animation"
    
    def __str__(self) -> str:
        """Enhanced string representation for Animation components."""
        parts = []
        if self.play_on_load:
            parts.append("autoPlay")

        # Show detailed clip information
        if self.clips:
            clip_details = []
            for clip in self.clips:
                targets = list(clip.curves.keys()) if clip.curves else []
                target_str = f", targets=[{', '.join(targets)}]" if targets else ""
                clip_details.append(f'"{clip.name}" ({clip.duration}s{target_str})')
            parts.append(f"clip={', '.join(clip_details)}")

        if self.default_clip:
            default_clip_name = next((clip.name for clip in self.clips if clip.name == self.default_clip), None)
            parts.append(f"default='{default_clip_name}'")

        if not parts:
            parts.append("empty")

        return f"Animation({', '.join(parts)})"

@dataclass
class WidgetComponent(DecodedComponent):
    """Decoded cc.Widget component."""
    align_flags: int = 0
    left: float = 0
    right: float = 0
    top: float = 0
    bottom: float = 0
    
    def __post_init__(self):
        self.component_type = "cc.Widget"
    
    def __str__(self) -> str:
        """Enhanced string representation for Widget components."""
        # Decode alignment flags
        flags = self.align_flags
        alignments = []
        if flags & 1: alignments.append("LEFT")
        if flags & 2: alignments.append("RIGHT")
        if flags & 4: alignments.append("TOP")
        if flags & 8: alignments.append("BOTTOM")
        if flags & 16: alignments.append("H_CENTER")
        if flags & 32: alignments.append("V_CENTER")
        
        align_str = "|".join(alignments) if alignments else "NONE"
        
        parts = [f"align={align_str}"]
        if self.left != 0: parts.append(f"left={self.left}")
        if self.right != 0: parts.append(f"right={self.right}")
        if self.top != 0: parts.append(f"top={self.top}")
        if self.bottom != 0: parts.append(f"bottom={self.bottom}")
        
        return f"Widget({', '.join(parts)})"

@dataclass
class LabelComponent(DecodedComponent):
    """Decoded cc.Label component."""
    text: str = ""
    font_size: int = 12
    font_asset: Optional[int] = None
    color: Any = None
    alignment: str = "left"
    
    def __post_init__(self):
        self.component_type = "cc.Label"
    
    def __str__(self) -> str:
        """XML-like string representation for Label components."""
        # Get horizontal and vertical alignment from properties
        h_align = "Left"
        v_align = "Top"
        
        # Check if we have alignment properties
        if hasattr(self, 'properties'):
            h_align_val = self.properties.get('_N$horizontalAlign', 0)
            v_align_val = self.properties.get('_N$verticalAlign', 0)
            
            # Map alignment values
            h_align_map = {0: "Left", 1: "Center", 2: "Right"}
            v_align_map = {0: "Top", 1: "Middle", 2: "Bottom"}
            
            h_align = h_align_map.get(h_align_val, "Left")
            v_align = v_align_map.get(v_align_val, "Top")
        
        # Build color string
        # Prefer explicit color field, fall back to stored properties
        color_str = "rgba(255,255,255,255)"  # Default white
        color_val = self.color
        if color_val is None and hasattr(self, "properties"):
            color_val = self.properties.get("_color") or self.properties.get("color")
        if color_val:
            color_str = str(color_val)
        
        parts = [f"text='{self.text}'"]
        if self.font_size:
            parts.append(f"fontSize={self.font_size}")
        parts.append(f"hAlign={h_align}")
        parts.append(f"vAlign={v_align}")
        parts.append(f"color={color_str}")
        
        return f"<Label {' '.join(parts)}>"

class AnimationDecoder(ComponentDecoder):
    """Decoder for cc.Animation components with enhanced clip parsing."""
    
    @property
    def component_type(self) -> str:
        return "cc.Animation"
    
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> AnimationComponent:
        component = AnimationComponent(
            component_type=self.component_type,
            template_index=raw_row[0] if raw_row else -1
        )
        
        # Animation properties mapping
        prop_mapping = {
            "playOnLoad": "play_on_load",
            "_clips": "clips",
            "_defaultClip": "default_clip"
        }
        
        # Decode properties from raw_row
        for i, value in enumerate(raw_row[1:], 1):  # Skip template index
            prop_name = prop_names[i-1] if i-1 < len(prop_names) else f"prop_{i-1}"
            
            # Map to component attribute if known
            if prop_name in prop_mapping:
                attr_name = prop_mapping[prop_name]
                if attr_name == "play_on_load" and isinstance(value, bool):
                    component.play_on_load = value
                elif attr_name == "clips" and isinstance(value, list):
                    # Parse animation clips - each clip might be an asset reference or embedded data
                    component.clips = self._parse_animation_clips(value, helper)
                elif attr_name == "default_clip" and isinstance(value, (str, int)):
                    if isinstance(value, str):
                        component.default_clip = value
                    elif helper.is_asset_reference(value):
                        component.add_asset_ref(value, "animation_clip")
            
            # Store all properties for completeness
            component.properties[prop_name] = helper.decode_value_type(value)
            
            # Check for asset references
            if helper.is_asset_reference(value):
                asset_type = helper.get_asset_type(value)
                component.add_asset_ref(value, asset_type)
        
        return component
    
    def _parse_animation_clips(self, clips_data: List[Any], helper: DecodeHelper) -> List[AnimationClip]:
        """Parse animation clips from various formats with comprehensive curve extraction."""
        clips = []
        
        for clip_data in clips_data:
            if isinstance(clip_data, dict):
                # Direct clip object with name, duration, curves
                clip = self._parse_clip_object(clip_data)
                if clip:
                    clips.append(clip)
            elif isinstance(clip_data, int) and helper.is_asset_reference(clip_data):
                # Asset reference to external clip
                clip_name = f"clip_{clip_data}"
                clips.append(AnimationClip(name=clip_name, duration=0.0))
                helper.asset_registry[clip_data] = {"class": "cc.AnimationClip", "name": clip_name}
            elif isinstance(clip_data, list) and len(clip_data) >= 2:
                # Embedded clip data - parse the complete format
                clip = self._parse_embedded_clip_data(clip_data)
                if clip:
                    clips.append(clip)
        
        return clips
    
    def _parse_embedded_clip_data(self, clip_data: List[Any]) -> Optional[AnimationClip]:
        """Parse embedded animation clip data with full curve information."""
        if len(clip_data) < 5:
            return None
        
        # Parse the structure: [template_idx, name, duration, wrapMode, sample, curveData]
        clip_name = clip_data[1] if len(clip_data) > 1 and isinstance(clip_data[1], str) else "unnamed"
        duration = clip_data[2] if len(clip_data) > 2 and isinstance(clip_data[2], (int, float)) else 1.0
        wrap_mode = clip_data[3] if len(clip_data) > 3 else "normal"
        sample_rate = clip_data[4] if len(clip_data) > 4 else 60
        
        clip = AnimationClip(name=clip_name, duration=float(duration), wrap_mode=str(wrap_mode))
        
        # Parse curve data if present (usually at index 5)
        if len(clip_data) > 5 and isinstance(clip_data[5], dict):
            clip.curves = self._parse_curves(clip_data[5])
        
        return clip
    
    def _parse_clip_object(self, clip_obj: Dict[str, Any]) -> Optional[AnimationClip]:
        """Parse a clip object from JSON format."""
        name = clip_obj.get("name", "unnamed")
        duration = clip_obj.get("duration", 1.0)
        wrap_mode = clip_obj.get("wrapMode", "normal")
        
        clip = AnimationClip(name=name, duration=float(duration), wrap_mode=wrap_mode)
        
        # Parse curve data if present
        if "curves" in clip_obj:
            clip.curves = self._parse_curves(clip_obj["curves"])
        elif "curveData" in clip_obj:
            clip.curves = self._parse_curves(clip_obj["curveData"])
        
        return clip
    
    def _parse_curves(self, curves_data: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
        """Parse animation curves for position, rotation, etc."""
        curves = {}
        
        # Handle nested curve structures - look for 'paths' containing the actual curve data
        if "paths" in curves_data:
            curves_data = curves_data["paths"]
        
        for node_name, node_data in curves_data.items():
            if isinstance(node_data, dict):
                curves[node_name] = {}
                
                # Handle nested props structure
                props = node_data.get("props", node_data)
                
                for prop_name, keyframes in props.items():
                    if isinstance(keyframes, list):
                        # Parse keyframe data - each keyframe has frame and value
                        parsed_keyframes = []
                        for kf in keyframes:
                            if isinstance(kf, dict) and "frame" in kf and "value" in kf:
                                parsed_keyframes.append({
                                    "time": kf["frame"],
                                    "value": kf["value"]
                                })
                            else:
                                parsed_keyframes.append(kf)
                        curves[node_name][prop_name] = parsed_keyframes
        
        return curves

class WidgetDecoder(ComponentDecoder):
    """Decoder for cc.Widget components."""
    
    @property
    def component_type(self) -> str:
        return "cc.Widget"
    
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> WidgetComponent:
        component = WidgetComponent(
            component_type=self.component_type,
            template_index=raw_row[0] if raw_row else -1
        )
        
        # Widget properties mapping
        prop_mapping = {
            "_alignFlags": "align_flags",
            "_left": "left",
            "_right": "right",
            "_top": "top",
            "_bottom": "bottom"
        }
        
        # Decode properties from raw_row
        for i, value in enumerate(raw_row[1:], 1):  # Skip template index
            prop_name = prop_names[i-1] if i-1 < len(prop_names) else f"prop_{i-1}"
            
            # Map to component attribute if known
            if prop_name in prop_mapping:
                attr_name = prop_mapping[prop_name]
                if attr_name == "align_flags" and isinstance(value, int):
                    component.align_flags = value
                elif attr_name in ["left", "right", "top", "bottom"] and isinstance(value, (int, float)):
                    setattr(component, attr_name, float(value))
            
            # Store all properties for completeness
            component.properties[prop_name] = helper.decode_value_type(value)
            
            # Check for asset references
            if helper.is_asset_reference(value):
                asset_type = helper.get_asset_type(value)
                component.add_asset_ref(value, asset_type)
        
        return component

class LabelDecoder(ComponentDecoder):
    """Decoder for cc.Label components."""
    
    @property
    def component_type(self) -> str:
        return "cc.Label"
    
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> LabelComponent:
        component = LabelComponent(
            component_type=self.component_type,
            template_index=raw_row[0] if raw_row else -1
        )
        
        # Common Label properties mapping
        prop_mapping = {
            "_string": "text",
            "_fontSize": "font_size", 
            "_N$file": "font_asset",
            "_N$horizontalAlign": "alignment"
        }
        
                # Try to extract text from various positions if property mapping fails
        text_found = False
        
        # Decode properties from raw_row
        for i, value in enumerate(raw_row[1:], 1):  # Skip template index
            prop_name = prop_names[i-1] if i-1 < len(prop_names) else f"prop_{i-1}"
            
            # Map to component attribute if known
            if prop_name in prop_mapping:
                attr_name = prop_mapping[prop_name]
                if attr_name == "text" and isinstance(value, str):
                    component.text = value
                    text_found = True
                elif attr_name == "font_size" and isinstance(value, (int, float)):
                    component.font_size = int(value)
                elif attr_name == "font_asset" and helper.is_asset_reference(value):
                    component.font_asset = value
                    component.add_asset_ref(value, "font")
                elif attr_name == "alignment":
                    component.alignment = self._decode_alignment(value)
            
            # Fallback: Look for string values in any position if text not found via property mapping
            elif not text_found and isinstance(value, str) and value.strip():
                component.text = value
                text_found = True
            
            # Store all properties for completeness
            component.properties[prop_name] = helper.decode_value_type(value)
            
            # Check for asset references
            if helper.is_asset_reference(value):
                asset_type = helper.get_asset_type(value)
                component.add_asset_ref(value, asset_type)
        
        return component
    
    def _decode_alignment(self, value: Any) -> str:
        """Decode alignment value to string."""
        if isinstance(value, int):
            alignment_map = {0: "left", 1: "center", 2: "right"}
            return alignment_map.get(value, "left")
        return str(value) if value else "left"

@dataclass
class SpriteComponent(DecodedComponent):
    """Decoded cc.Sprite component."""
    sprite_frame: Optional[int] = None
    size_mode: str = "trimmed"
    sprite_type: str = "simple"
    fill_range: float = 1.0
    
    def __post_init__(self):
        self.component_type = "cc.Sprite"
    
    def __str__(self) -> str:
        """XML-like string representation for Sprite components with sprite frame resolution."""
        parts = []

        # Handle sprite frame reference with name resolution
        sprite_frame_display = "SpriteFrame#?"

        # Check if resolved frame information was set by TabRow.from_raw
        if hasattr(self, '_resolved_frame_name') and self._resolved_frame_name:
            frame_name = self._resolved_frame_name
            sprite_frame_display = f'"{frame_name}"'
            if hasattr(self, '_resolved_frame_data') and self._resolved_frame_data:
                frame_data = self._resolved_frame_data
                width = frame_data.get('width', 0)
                height = frame_data.get('height', 0)
                sprite_frame_display += f' ({width}×{height})'
        elif self.sprite_frame:
            # Fallback: show asset reference
            sprite_frame_display = f"SpriteFrame#{self.sprite_frame}"

        parts.append(f"spriteFrame={sprite_frame_display}")
        
        # Map size mode to expected format
        size_mode_map = {"trimmed": "SIMPLE", "raw": "RAW", "custom": "CUSTOM"}
        size_mode_display = size_mode_map.get(self.size_mode, "SIMPLE")
        parts.append(f"sizeMode={size_mode_display}")
        
        return f"<Sprite {' '.join(parts)}>"

@dataclass
class ButtonComponent(DecodedComponent):
    """Decoded cc.Button component."""
    transition_type: str = "none"
    target_node: Optional[int] = None
    normal_color: Any = None
    pressed_color: Any = None
    disabled_color: Any = None
    zoom_scale: float = 1.1
    
    def __post_init__(self):
        self.component_type = "cc.Button"
    
    def __str__(self) -> str:
        """Enhanced string representation for Button components."""
        parts = []
        if self.transition_type != "none":
            parts.append(f"transition={self.transition_type}")
        if self.zoom_scale != 1.1:
            parts.append(f"zoomScale={self.zoom_scale}")
        if self.target_node:
            parts.append(f"target={self.target_node}")
        
        if not parts:
            parts.append("default")
        
        asset_str = f" [assets: {len(self.asset_refs)}]" if self.asset_refs else ""
        return f"{self.component_type}({', '.join(parts)}){asset_str}"

class SpriteDecoder(ComponentDecoder):
    """Decoder for cc.Sprite components."""
    
    @property  
    def component_type(self) -> str:
        return "cc.Sprite"
    
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> SpriteComponent:
        component = SpriteComponent(
            component_type=self.component_type,
            template_index=raw_row[0] if raw_row else -1
        )
        
        # Common Sprite properties mapping
        prop_mapping = {
            "_spriteFrame": "sprite_frame",
            "_sizeMode": "size_mode",
            "_type": "sprite_type",
            "_fillRange": "fill_range"
        }
        
        # Decode properties from raw_row
        for i, value in enumerate(raw_row[1:], 1):  # Skip template index
            prop_name = prop_names[i-1] if i-1 < len(prop_names) else f"prop_{i-1}"
            
            # Map to component attribute if known
            if prop_name in prop_mapping:
                attr_name = prop_mapping[prop_name]
                if attr_name == "sprite_frame" and helper.is_asset_reference(value):
                    component.sprite_frame = value
                    component.add_asset_ref(value, "sprite_frame")
                elif attr_name == "size_mode" and isinstance(value, int):
                    component.size_mode = self._decode_size_mode(value)
                elif attr_name == "sprite_type" and isinstance(value, int):
                    component.sprite_type = self._decode_sprite_type(value)
                elif attr_name == "fill_range" and isinstance(value, (int, float)):
                    component.fill_range = float(value)
            
            # Store all properties for completeness
            component.properties[prop_name] = helper.decode_value_type(value)
            
            # Check for asset references
            if helper.is_asset_reference(value):
                asset_type = helper.get_asset_type(value)
                component.add_asset_ref(value, asset_type)
        
        return component
    
    def _decode_size_mode(self, value: int) -> str:
        """Decode size mode value to string."""
        size_modes = {0: "trimmed", 1: "raw", 2: "custom"}
        return size_modes.get(value, "trimmed")
    
    def _decode_sprite_type(self, value: int) -> str:
        """Decode sprite type value to string.""" 
        sprite_types = {0: "simple", 1: "sliced", 2: "tiled", 3: "filled"}
        return sprite_types.get(value, "simple")

# ───────────────────────────── Fallback Decoder ───────────────────────────────

class FallbackDecoder(ComponentDecoder):
    """Fallback decoder for unknown component types."""
    
    def __init__(self, component_type: str):
        self._component_type = component_type
    
    @property
    def component_type(self) -> str:
        return self._component_type
    
    def decode(self, raw_row: List[Any], prop_names: List[str], helper: DecodeHelper) -> DecodedComponent:
        """
        Fallback decoder that iterates all fields, applies value-type/asset detection,
        and stores a {prop: value} dict so unknown components are still readable.
        """
        component = DecodedComponent(
            component_type=self.component_type,
            template_index=raw_row[0] if raw_row else -1
        )
        
        # Process all properties with type detection
        for i, value in enumerate(raw_row[1:], 1):  # Skip template index
            prop_name = prop_names[i-1] if i-1 < len(prop_names) else f"prop_{i-1}"
            
            # Apply value type decoding
            decoded_value = helper.decode_value_type(value)
            component.properties[prop_name] = decoded_value
            
            # Apply asset detection
            if helper.is_asset_reference(value):
                asset_type = helper.get_asset_type(value)
                component.add_asset_ref(value, asset_type)
            
            # Recursive asset detection for nested structures
            self._detect_nested_assets(decoded_value, component, helper)
        
        return component
    
    def _detect_nested_assets(self, value: Any, component: DecodedComponent, helper: DecodeHelper):
        """Recursively detect asset references in nested structures."""
        if isinstance(value, list):
            for item in value:
                if helper.is_asset_reference(item):
                    asset_type = helper.get_asset_type(item)
                    component.add_asset_ref(item, asset_type)
                else:
                    self._detect_nested_assets(item, component, helper)
        elif isinstance(value, dict):
            for item in value.values():
                if helper.is_asset_reference(item):
                    asset_type = helper.get_asset_type(item)
                    component.add_asset_ref(item, asset_type)
                else:
                    self._detect_nested_assets(item, component, helper)

# ───────────────────────────── Decoder Registry ───────────────────────────────

# Registry mapping component types to their decoders
DECODERS: Dict[str, Type[ComponentDecoder]] = {
    "cc.Label": LabelDecoder,
    "cc.Sprite": SpriteDecoder,
    "cc.Animation": AnimationDecoder,
    "cc.Widget": WidgetDecoder,
    # Add more decoders here as needed
}

class ComponentDecoderEngine:
    """Main engine for decoding components using registered decoders.
    
    Provides a registry-based system for converting packed component data
    into structured Python objects. Each component type (cc.Label, cc.Sprite, etc.)
    can have a specialized decoder that understands its property layout.
    
    The engine automatically falls back to a generic decoder for unknown types,
    ensuring all components can be processed even without explicit support.
    """
    
    def __init__(self, decoders: Optional[Dict[str, Type[ComponentDecoder]]] = None) -> None:
        """Initialize the decoder engine.
        
        Args:
            decoders: Dictionary mapping component type names to decoder classes.
                     If None, uses the default DECODERS registry.
        """
        self.decoders = decoders or DECODERS.copy()
        self._decoder_instances: Dict[str, ComponentDecoder] = {}
    
    def register_decoder(self, component_type: str, decoder_class: Type[ComponentDecoder]):
        """Register a new decoder for a component type."""
        self.decoders[component_type] = decoder_class
        # Clear cached instance if it exists
        if component_type in self._decoder_instances:
            del self._decoder_instances[component_type]
    
    def get_decoder(self, component_type: str) -> ComponentDecoder:
        """Get decoder instance for a component type, creating fallback if needed."""
        if component_type not in self._decoder_instances:
            if component_type in self.decoders:
                self._decoder_instances[component_type] = self.decoders[component_type]()
            else:
                # Create fallback decoder for unknown types
                self._decoder_instances[component_type] = FallbackDecoder(component_type)
        
        return self._decoder_instances[component_type]
    
    def decode_component(self, 
                        component_type: str,
                        raw_row: List[Any], 
                        prop_names: List[str], 
                        helper: DecodeHelper) -> DecodedComponent:
        """
        Decode a component using the appropriate decoder.
        """
        decoder = self.get_decoder(component_type)
        return decoder.decode(raw_row, prop_names, helper)
    
    def get_supported_types(self) -> List[str]:
        """Get list of explicitly supported component types."""
        return list(self.decoders.keys())

# ───────────────────────────── component models ───────────────────────────────
@dataclass
class Component:
    tpl: int
    cls: str
    properties: Dict[str, Any] = field(default_factory=dict)
    decoded_component: Optional[DecodedComponent] = None
    
    def __str__(self) -> str:
        if self.decoded_component:
            # Use the decoded component's enhanced __str__ method
            return str(self.decoded_component)
        elif self.cls == "cc.Sprite":
            # Fallback XML format for Sprite components without decoded_component
            if self.decoded_component:
                frame_display = getattr(self.decoded_component, '_resolved_frame_name', 'UNKNOWN')
                if frame_display != 'UNKNOWN' and hasattr(self.decoded_component, '_resolved_frame_data'):
                    frame_data = self.decoded_component._resolved_frame_data
                    width = frame_data.get('width', 0)
                    height = frame_data.get('height', 0)
                    return f"<Sprite spriteFrame=\"{frame_display}\" ({width}×{height}) sizeMode=SIMPLE>"
            return f"<Sprite spriteFrame=SpriteFrame#? sizeMode=SIMPLE>"
        elif self.cls == "cc.Label" and hasattr(self, 'text'):
            # Fallback for Label components
            return f"<Label text='{getattr(self, 'text', '')}' fontSize=45 hAlign=Center vAlign=Middle color=rgba(255,255,255,255)>"
        elif self.properties:
            prop_strs = [f"{k}={v}" for k, v in self.properties.items() if v is not None]
            return f"{self.cls}({', '.join(prop_strs[:3])}...)"  # Limit to first 3 props
        return f"{self.cls}(tpl={self.tpl})"

class LabelComp(Component):
    def __init__(self, tpl: int, cls: str, text: str, font_tpl: int | None, color_tpl: int | None):
        super().__init__(tpl, cls)
        self.text = text
        self.font_tpl = font_tpl
        self.color_tpl = color_tpl
    
    def __str__(self) -> str:
        # XML-like format to match the expected output
        parts = [f"text='{self.text}'"]
        parts.append("fontSize=45")  # Default font size for this format
        parts.append("hAlign=Center")
        parts.append("vAlign=Middle")
        parts.append("color=rgba(255,255,255,255)")
        return f"<Label {' '.join(parts)}>"

# ────────────────────────────── Tab-row wrapper ───────────────────────────────
class TabRow:
    """Decode one packed node instance from pack data.
    
    Represents a single node in the scene graph, decoded from packed format.
    Handles extraction of transform data, components, and hierarchy information.
    
    The packed format encodes nodes as arrays where each position has semantic meaning:
    [0] = template_index, [1] = name, [2] = parent_index, [3] = components, etc.
    """

    def __init__(self,
                 tpl: int,
                 name: Optional[str],
                 parent: Optional[int],
                 size: str,
                 pos: str, rot: str, scale: str,
                 comps: List[Component],
                 raw: List[Any]) -> None:
        """Initialize a decoded node row.
        
        Args:
            tpl: Template index for this node type
            name: Node name (if any)
            parent: Parent node index (if any)
            size: Size string representation (e.g., "(100×50)")
            pos: Position string (e.g., "(10,20,0)")
            rot: Rotation string (e.g., "(0,0,45)")
            scale: Scale string (e.g., "(1,1,1)")
            comps: List of decoded components attached to this node
            raw: Original raw packed data for debugging
        """
        self.tpl, self.name, self.parent = tpl, name, parent
        self.size, self.pos, self.rot, self.scale = size, pos, rot, scale
        self.components = comps
        self.raw = raw

    @classmethod
 codex/modify-tabrow-to-accept-decoder_engine
    def from_raw(cls,
                 node_row: List[Any],
                 templates: List[Any],
                 class_of,
                 pack_format: Optional[PackFormatInfo] = None,
                 asset_registry: Dict[int, Any] = None,
                 bundle_instance: Optional['CocosBundle'] = None,
                 decoder_engine: Optional[ComponentDecoderEngine] = None) -> "TabRow":
        """Create a :class:`TabRow` from a raw packed node entry.

        Parameters
        ----------
        node_row:
            Raw list from the pack representing a node.
        templates:
            Template table from the pack data.
        class_of:
            Function mapping a template index to its class name.
        pack_format:
            Parsed :class:`PackFormatInfo` describing the pack (optional).
        asset_registry:
            Mapping of asset indices to metadata (optional).
        bundle_instance:
            Reference to the :class:`CocosBundle` being processed (optional).
        decoder_engine:
            Shared :class:`ComponentDecoderEngine` instance to decode
            components. If ``None`` a new engine is created.
        """

    def from_raw(cls,
                 node_row: List[Any],
                 templates: List[Any],
                 class_of,
                 pack_format: Optional[PackFormatInfo] = None,
                 asset_registry: Dict[int, Any] = None,
                 bundle_instance: Optional['CocosBundle'] = None,
                 node_obj: Optional[Dict[str, Any]] = None) -> "TabRow":
        main
        tpl_index = node_row[0]
        node_name = node_row[1] if len(node_row) > 1 and isinstance(node_row[1], str) else None
        parent_index = node_row[2] if len(node_row) > 2 and isinstance(node_row[2], int) else None

        # Size extraction - be very conservative to avoid using position data
        size = "-"
        
        # Only try IDX_SIZE position and value-type arrays
        if len(node_row) > IDX_SIZE and isinstance(node_row[IDX_SIZE], list):
            arr = node_row[IDX_SIZE]
            # Only accept proper value-type arrays for Size
            if ValueTypeDeserializer.is_value_array(arr):
                val = ValueTypeDeserializer.decode(arr)
                if isinstance(val, Size):
                    size = str(val)
            # Very restrictive check for direct size data: must be exactly 2 positive numbers in a small range
            elif (len(arr) == 2 and 
                  all(isinstance(x, (int, float)) for x in arr) and
                  all(x > 0 and x < 5000 for x in arr)):  # Reasonable UI element size range
                try:
                    size = f"({_num(arr[0])}×{_num(arr[1])})"
                except (IndexError, TypeError):
                    pass
        
        # Do NOT try to extract size from other positions as it often picks up position data

        # Transform - try different indices for transform data
        pos = rot = scale = "-"
        trs_data = None
        
        # Try to find transform data in different positions
        for idx in [IDX_TRS, IDX_TRS-1, IDX_TRS+1, -1]:  # Last item sometimes contains transforms
            if (idx == -1 and len(node_row) > 0) or (0 <= idx < len(node_row)):
                candidate = node_row[idx] if idx != -1 else node_row[-1]
                if isinstance(candidate, list) and len(candidate) >= 9:
                    # Check if this looks like transform data (numbers)
                    if all(isinstance(x, (int, float)) for x in candidate[:9]):
                        trs_data = candidate
                        break
        
        if trs_data:
            # Extract position, rotation, scale from transform data
            px, py, pz = trs_data[0], trs_data[1], trs_data[2]
            rx, ry, rz = trs_data[3], trs_data[4], trs_data[5]
            sx, sy, sz = trs_data[6], trs_data[7], trs_data[8]
            
            # Always show position if any component is non-zero
            if px != 0 or py != 0:
                pos = f"({_num(px)},{_num(py)},)"
            if rx != 0 or ry != 0 or rz != 0:
                rot = f"({_num(rx)},{_num(ry)},{_num(rz)})"
            if sx != 1 or sy != 1 or sz != 1:
                scale = f"({_num(sx)},{_num(sy)},{_num(sz)})"

        #print(node_row)
        # Components with enhanced property mapping using ComponentDecoder engine
        components: List[Component] = []
        decoder_engine = decoder_engine or ComponentDecoderEngine()
        
        if len(node_row) > IDX_OVR and isinstance(node_row[IDX_OVR], list):
            for comp_row in node_row[IDX_OVR]:
                if not (isinstance(comp_row, list) and comp_row and isinstance(comp_row[0], int)):
                    continue
                comp_tpl = comp_row[0]
                comp_cls = class_of(templates[comp_tpl][0]) if 0 <= comp_tpl < len(templates) else "Unknown"
                
                # Debug: Show what components are being processed
                # if comp_cls in ["cc.Sprite", "cc.Animation"]:
                #     print(f"      DEBUG: Processing {comp_cls} component for node '{node_name}'")
                
                # Build DecodeHelper with context
                template_info = None
                prop_names = []
                asset_registry = getattr(bundle_instance, 'assets', {}) if bundle_instance else {}
                
                if pack_format and comp_tpl in pack_format.class_templates:
                    template_info = pack_format.class_templates[comp_tpl]
                    prop_names = template_info.properties
                
                helper = DecodeHelper(
                    template_info=template_info,
                    asset_registry=asset_registry,
                    property_names=prop_names,
                    value_type_decoder=ValueTypeDeserializer
                )
                
                # Add the class method for asset detection with proper asset registry parameter
                def is_asset_ref(value: int) -> bool:
                    return (isinstance(value, int) and
                            (value in asset_registry or
                             value > ASSET_REF_THRESHOLD))  # Fallback heuristic for large indices
                helper.is_asset_reference = is_asset_ref
                
                # Enhanced asset type detection
                def get_enhanced_asset_type(asset_id: int) -> str:
                    if asset_id in asset_registry:
                        asset_info = asset_registry[asset_id]
                        if isinstance(asset_info, dict) and 'class' in asset_info:
                            return asset_info['class']
                    
                    return "unknown"
                
                helper.get_asset_type = get_enhanced_asset_type
                
                # Use ComponentDecoder engine to decode the component
                try:
                    # Pass bundle context to decoders for sprite frame resolution
                    if hasattr(self, 'sprite_frames'):
                        helper.sprite_frames = self.sprite_frames
                    if hasattr(self, 'embedded_animations'):
                        helper.embedded_animations = self.embedded_animations
                    
                    # Store bundle sprite frames for later use
                    if bundle_instance and hasattr(bundle_instance, 'sprite_frames'):
                        helper.sprite_frames = bundle_instance.sprite_frames
                    
                    # Enhanced sprite frame resolution for components
                    def resolve_sprite_frame_name(asset_id: int) -> str:
                        # Try to match sprite frame by node name if asset_id doesn't match directly
                        current_node_name = node_name if node_name else None
                        
                        # Check if we have sprite frames registry
                        if hasattr(helper, 'sprite_frames') and helper.sprite_frames:
                            sprite_frames = helper.sprite_frames
                            # First try: match by node name
                            if current_node_name and current_node_name in sprite_frames:
                                frame_data = sprite_frames[current_node_name]
                                width = frame_data.get('width', 0)
                                height = frame_data.get('height', 0)
                                return f'"{current_node_name}" ({width}×{height})'
                            
                            # Second try: if there's only one or two sprite frames, use them by process of elimination
                            sprite_frame_names = list(sprite_frames.keys())
                            if len(sprite_frame_names) == 1:
                                frame_name = sprite_frame_names[0]
                                frame_data = sprite_frames[frame_name]
                                width = frame_data.get('width', 0)
                                height = frame_data.get('height', 0)
                                return f'"{frame_name}" ({width}×{height})'
                            elif len(sprite_frame_names) == 2:
                                # Use heuristics - white_loading is usually the small spinning loader, GameIcon is large
                                for frame_name in sprite_frame_names:
                                    if current_node_name and frame_name.lower() in current_node_name.lower():
                                        frame_data = sprite_frames[frame_name]
                                        width = frame_data.get('width', 0)
                                        height = frame_data.get('height', 0)
                                        return f'"{frame_name}" ({width}×{height})'
                                
                                # Fallback: match by size - if node is white_loading, use smaller frame; if GameIcon, use larger
                                if current_node_name:
                                    if 'white_loading' in current_node_name.lower():
                                        # Use the smaller sprite frame
                                        smaller_frame = min(sprite_frame_names, key=lambda f: sprite_frames[f]['width'])
                                        frame_data = sprite_frames[smaller_frame]
                                        width = frame_data.get('width', 0)
                                        height = frame_data.get('height', 0)
                                        return f'"{smaller_frame}" ({width}×{height})'
                                    elif 'gameicon' in current_node_name.lower():
                                        # Use the larger sprite frame
                                        larger_frame = max(sprite_frame_names, key=lambda f: sprite_frames[f]['width'])
                                        frame_data = sprite_frames[larger_frame]
                                        width = frame_data.get('width', 0)
                                        height = frame_data.get('height', 0)
                                        return f'"{larger_frame}" ({width}×{height})'
                        
                        return f"SpriteFrame#{asset_id}"
                    
                    helper.resolve_sprite_frame_name = resolve_sprite_frame_name
                    
                    # Enhanced asset reference detection for sprite frames
                    def enhanced_asset_detection(value: int) -> bool:
                        if not isinstance(value, int):
                            return False
                        # Check if it's in our asset registry
                        if value in asset_registry:
                            return True
                        # For sprite components, be more liberal with asset detection
                        if comp_cls == "cc.Sprite" and abs(value) > 0:
                            return True
                        # Standard detection for other components
                        return abs(value) > ASSET_REF_THRESHOLD
                    helper.is_asset_reference = enhanced_asset_detection
                    
                    decoded_component = decoder_engine.decode_component(
                        component_type=comp_cls,
                        raw_row=comp_row,
                        prop_names=prop_names,
                        helper=helper
                    )
                    
                    # Create the component with decoded information
                    if comp_cls == "cc.Label" and isinstance(decoded_component, LabelComponent):
                        # Special handling for Labels - keep backward compatibility
                        text = decoded_component.text
                        font = decoded_component.font_asset

                        # Extract node color from object or component properties
                        # Node _color values map directly to our Color dataclass
                        def _decode_color(val: Any) -> Any:
                            if val is None:
                                return None
                            # Support value-type arrays, dicts or packed ints
                            if isinstance(val, list):
                                if ValueTypeDeserializer.is_value_array(val):
                                    val = ValueTypeDeserializer.decode(val)
                                    if isinstance(val, Color):
                                        return val
                                if len(val) in (3, 4) and all(isinstance(x, (int, float)) for x in val):
                                    r, g, b = int(val[0]), int(val[1]), int(val[2])
                                    a = int(val[3]) if len(val) == 4 else 255
                                    return Color(r, g, b, a)
                            if isinstance(val, dict):
                                r = int(val.get('r', 255))
                                g = int(val.get('g', 255))
                                b = int(val.get('b', 255))
                                a = int(val.get('a', 255))
                                return Color(r, g, b, a)
                            if isinstance(val, int):
                                decoded = ValueTypeDeserializer.decode([ValueTypeID.Color, val])
                                if isinstance(decoded, Color):
                                    return decoded
                            if isinstance(val, Color):
                                return val
                            return None

                        col = None
                        if node_obj and isinstance(node_obj, dict):
                            col = _decode_color(node_obj.get('_color'))
                        if not col and hasattr(decoded_component, 'properties'):
                            col = _decode_color(
                                decoded_component.properties.get('_color') or
                                decoded_component.properties.get('color'))

                        comp = LabelComp(comp_tpl, comp_cls, text, font, col)
                        comp.decoded_component = decoded_component
                        comp.properties = decoded_component.properties
                        decoded_component.color = col
                        components.append(comp)
                    elif comp_cls == "cc.Sprite" and isinstance(decoded_component, SpriteComponent):
                        # Enhanced sprite frame resolution using comprehensive matching strategies
                        if bundle_instance and hasattr(bundle_instance, 'sprite_frames'):
                            sprite_frames = bundle_instance.sprite_frames
                            matched_frame = None
                            
                            # Strategy 1: Exact node name match
                            if node_name and node_name in sprite_frames:
                                matched_frame = node_name
                            
                            # Strategy 2: Use sprite frame asset reference if it maps to a known frame
                            elif decoded_component.sprite_frame and hasattr(bundle_instance, 'asset_uuid_map'):
                                sprite_frame_ref = decoded_component.sprite_frame
                                # Try to resolve sprite frame reference through UUID mapping
                                if sprite_frame_ref in bundle_instance.asset_uuid_map:
                                    uuid_name = bundle_instance.asset_uuid_map[sprite_frame_ref]
                                    if uuid_name in sprite_frames:
                                        matched_frame = uuid_name
                            
                            # Strategy 3: Smart fuzzy matching by node name
                            if not matched_frame and node_name:
                                sprite_frame_names = list(sprite_frames.keys())
                                
                                # Try exact substring matching (case insensitive)
                                for frame_name in sprite_frame_names:
                                    if (frame_name.lower() in node_name.lower()) or \
                                       (node_name.lower() in frame_name.lower()):
                                        matched_frame = frame_name
                                        break
                                
                                # Try common name pattern matching
                                if not matched_frame:
                                    # Common patterns: GameIcon/game_icon, white_loading/white-loading, etc.
                                    normalized_node = node_name.lower().replace('_', '').replace('-', '')
                                    for frame_name in sprite_frame_names:
                                        normalized_frame = frame_name.lower().replace('_', '').replace('-', '')
                                        if normalized_node == normalized_frame or \
                                           normalized_frame in normalized_node or \
                                           normalized_node in normalized_frame:
                                            matched_frame = frame_name
                                            break
                            
                            # Strategy 4: Size-based heuristics for well-known component types
                            if not matched_frame:
                                sprite_frame_names = list(sprite_frames.keys())
                                if len(sprite_frame_names) >= 2:
                                    if node_name and ('white_loading' in node_name.lower() or 'loading' in node_name.lower()):
                                        # Use smaller frame for loading components
                                        matched_frame = min(sprite_frame_names, key=lambda f: sprite_frames[f]['width'])
                                    elif node_name and ('gameicon' in node_name.lower() or 'icon' in node_name.lower()):
                                        # Use larger frame for icons
                                        matched_frame = max(sprite_frame_names, key=lambda f: sprite_frames[f]['width'])
                            
                            # Strategy 5: Process of elimination based on available frames
                            if not matched_frame:
                                sprite_frame_names = list(sprite_frames.keys())
                                if len(sprite_frame_names) == 1:
                                    # Only one sprite frame available, use it
                                    matched_frame = sprite_frame_names[0]
                                elif len(sprite_frame_names) == 2:
                                    # Two frames: usually white_loading (small) and GameIcon (large)
                                    if 'white_loading' in sprite_frame_names and 'GameIcon' in sprite_frame_names:
                                        # Default heuristic: if node name suggests loading, use white_loading; otherwise GameIcon
                                        if node_name and ('loading' in node_name.lower() or 'white' in node_name.lower()):
                                            matched_frame = 'white_loading'
                                        else:
                                            matched_frame = 'GameIcon'
                                    else:
                                        # For other combinations, use the first frame as fallback
                                        matched_frame = sprite_frame_names[0]
                            
                            # Apply the matched frame with full resolution
                            if matched_frame and matched_frame in sprite_frames:
                                frame_data = sprite_frames[matched_frame]
                                decoded_component._resolved_frame_name = matched_frame
                                decoded_component._resolved_frame_data = frame_data
                                # Ensure the frame information is directly used in display
                                comp_string = f"<Sprite spriteFrame=\"{matched_frame}\" ({frame_data['width']}×{frame_data['height']}) sizeMode=SIMPLE>"
                                decoded_component.__str__ = lambda: comp_string
                        
                        # Apply sprite frame resolution before creating Component
                        decoded_component._resolved_frame_name = getattr(decoded_component, '_resolved_frame_name', None)
                        decoded_component._resolved_frame_data = getattr(decoded_component, '_resolved_frame_data', None)
                        
                        comp = Component(
                            tpl=comp_tpl,
                            cls=comp_cls,
                            properties=decoded_component.properties,
                            decoded_component=decoded_component
                        )
                        components.append(comp)
                    else:
                        # Generic component with decoded properties
                        comp = Component(
                            tpl=comp_tpl,
                            cls=comp_cls,
                            properties=decoded_component.properties,
                            decoded_component=decoded_component
                        )
                        components.append(comp)
                        
                except Exception as e:
                    # Fallback to original property mapping if decoder fails
                    properties = {}
                    if pack_format and comp_tpl in pack_format.class_templates:
                        template_info = pack_format.class_templates[comp_tpl]
                        for i, value in enumerate(comp_row[1:]):
                            if i < len(template_info.properties):
                                prop_name = template_info.properties[i]
                                properties[prop_name] = value
                            else:
                                properties[f"prop_{i}"] = value
                    
                    # Legacy component creation as fallback
                    if comp_cls == "cc.Label":
                        # Try multiple positions for text extraction
                        text = ""
                        for i in range(1, min(len(comp_row), 6)):  # Check first few positions for string
                            if isinstance(comp_row[i], str) and comp_row[i].strip():
                                text = comp_row[i]
                                break
                        
                        font = comp_row[6] if len(comp_row) > 6 and isinstance(comp_row[6], int) else None
                        col  = comp_row[7][0] if len(comp_row) > 7 and isinstance(comp_row[7], list) and comp_row[7] and isinstance(comp_row[7][0], int) else None
                        comp = LabelComp(comp_tpl, comp_cls, text, font, col)
                        comp.properties = properties
                        components.append(comp)
                    else:
                        components.append(Component(comp_tpl, comp_cls, properties))

        return cls(tpl_index, node_name, parent_index, size, pos, rot, scale, components, node_row)

# ───────────────────────── scene / asset dataclasses ─────────────────────────
@dataclass
class AssetInfo:
    idx:int; cls:str; name:str|None; path:str
    def row(self)->str:
        n = f"'{self.name}'" if self.name else "(no name)"
        return f"[{self.idx:>3}] {self.cls:18} {n:28} {self.path}"

@dataclass
class Node:
    name:str; cls:str; key:Tuple[int,int]
    pos:str="-"; rot:str="-"; scale:str="-"; anc:str="-"; size:str="-"
    comps:List[Any]=field(default_factory=list)
    children:List["Node"]=field(default_factory=list)
    assets:Set[int]=field(default_factory=set)
    def __hash__(self): return hash(self.key)
    def __eq__(self,o): return isinstance(o,Node) and o.key==self.key
    def add_child(self,c:"Node"): 
        if c not in self.children: self.children.append(c)

# ───────────────────────────── Bundle inspector ──────────────────────────────
class CocosBundle:
    """Main inspector class for Cocos Creator bundle/pack analysis.
    
    This class orchestrates the entire inspection process:
    1. Loads bundle configuration JSON
    2. Discovers and processes pack files
    3. Extracts scene graphs, components, and assets
    4. Provides formatted output with hierarchy visualization
    
    Pack File Format Constants:
        IDX_CLASS: Index of class definitions in pack data (3)
        IDX_TPL: Index of templates in pack data (4)
        IDX_FIRST: Index where actual node data begins (5)
    """
    IDX_CLASS, IDX_TPL, IDX_FIRST = 3, 4, 5
    
    def __init__(self, cfg: Union[str, Path]) -> None:
        """Initialize bundle inspector from configuration file.
        
        Args:
            cfg: Path to bundle configuration JSON file
            
        The configuration file typically contains:
        - importBase: Base directory for import files
        - paths: Asset path mappings
        - packs: Pack file references
        - versions: Version mappings for pack files
        """
        self.cfg_path = Path(cfg).resolve()
        self.cfg      = json.loads(self.cfg_path.read_text("utf8"))
        self.root     = self.cfg_path.parent
        self.import_base = self.cfg.get("importBase","import")
        self.paths: Dict[str, Any] = self.cfg.get("paths", {})
        self.packs: Dict[str, list] = self.cfg.get("packs", {})
        self.ver = self._ver_map()
        self.pack_formats: Dict[str, PackFormatInfo] = {}  # Cache for pack format info
        self.show_full_assets = False  # Show only referenced assets by default

    def run(self):  # CLI entry
        for label, p in self._packs(): self._inspect_pack(label,p)

    def _ver_map(self)->Dict[str,str]:
        raw = self.cfg.get("versions", {}).get("import", [])
        if isinstance(raw, dict): return {k:str(v) for k,v in raw.items()}
        return {raw[i]: str(raw[i+1]) for i in range(0, len(raw), 2)}
    def _packs(self)->List[Tuple[str,Path]]:
        seen=set()
        for pk in self.packs:
            ver=self.ver.get(pk)
            path=self.root/self.import_base/pk[:2]/f"{pk}.{ver}.json"
            if ver and path.exists() and path not in seen:
                seen.add(path); yield f"pack:{pk}", path

    def _inspect_pack(self,label:str,pack:Path):
        """Inspect a single pack file and decode its contents.

        A :class:`ComponentDecoderEngine` is created once per pack and
        passed to all :meth:`TabRow.from_raw` calls so component decoders
        can share state across nodes.
        """
        data=json.loads(pack.read_text("utf8"))
        if not (isinstance(data,list) and len(data)>self.IDX_FIRST):
            return print(f"[skip] {label}")
        print(f"\n=== {label} ===")
        
        # Extract and store embedded animation clips from the first data block
        self._extract_embedded_animation_clips(data)
        
        # Extract sprite frame data and asset information
        self._extract_sprite_frames_and_assets(data)

        # Enhanced pack format analysis
        try:
            pack_format = PackFormatInfo.from_pack_data(data)
            self.pack_formats[label] = pack_format
            print(f"    Pack format: v{pack_format.format_version}, {pack_format.uuids_count} UUIDs, {len(pack_format.property_names)} properties")
            print(f"    Class templates: {len(pack_format.class_templates)} mapped")
            
            # Print unique classes found
            unique_classes = set(template.class_name for template in pack_format.class_templates.values())
            print(f"    Unique classes: {', '.join(sorted(unique_classes))}")
        except Exception as e:
            print(f"    Warning: Could not parse pack format: {e}")
            pack_format = None

        class_defs=data[self.IDX_CLASS]; templates=data[self.IDX_TPL]
        blocks=self._blocks(data[self.IDX_FIRST:])
        class_of=lambda i:class_defs[i][0] if 0<=i<len(class_defs) and isinstance(class_defs[i],list) else f"C?{i}"
        assets=self._asset_index(templates,class_of)
        offs=self._block_offs(blocks)
        block_id=self._scene_block(blocks,templates,class_of); rows=blocks[block_id]

 codex/modify-tabrow-to-accept-decoder_engine
        decoder_engine = ComponentDecoderEngine()
        nodes:Dict[Tuple[int,int],Node]={}
        for i, raw in enumerate(rows):
            # Pass the bundle instance to TabRow for sprite frame resolution
            row = TabRow.from_raw(
                raw,
                templates,
                class_of,
                pack_format,
                asset_registry=getattr(self, 'assets', {}),
                bundle_instance=self,
                decoder_engine=decoder_engine,
            )
            g_idx = offs[block_id]+i
            obj = data[g_idx] if 0<=g_idx<len(data) and isinstance(data[g_idx],dict) else {}

        nodes:Dict[Tuple[int,int],Node]={}
        for i, raw in enumerate(rows):
            g_idx = offs[block_id]+i
            obj = data[g_idx] if 0<=g_idx<len(data) and isinstance(data[g_idx],dict) else {}
            # Pass the bundle instance and node object to TabRow
            row = TabRow.from_raw(
                raw, templates, class_of, pack_format,
                asset_registry=getattr(self, 'assets', {}),
                bundle_instance=self,
                node_obj=obj,
            )
 main
            # Enhanced size extraction: try object block's "_contentSize" if not found in raw_row
            enhanced_size = row.size
            if enhanced_size == "-" and "_contentSize" in obj:
                content_size = obj["_contentSize"]
                if isinstance(content_size, dict) and "width" in content_size and "height" in content_size:
                    # Dict format: {"width": w, "height": h}
                    w, h = content_size["width"], content_size["height"]
                    enhanced_size = f"({_num(w)}×{_num(h)})"
                elif isinstance(content_size, list) and len(content_size) >= 2:
                    # List format: [w, h] or value-type array
                    if ValueTypeDeserializer.is_value_array(content_size):
                        val = ValueTypeDeserializer.decode(content_size)
                        if isinstance(val, Size):
                            enhanced_size = str(val)  # This gives us "(w×h)" format
                    else:
                        # Simple list [w, h]
                        try:
                            enhanced_size = f"({_num(content_size[0])}×{_num(content_size[1])})"
                        except (IndexError, TypeError):
                            pass
            
            # Completely override incorrect size data that matches position
            if enhanced_size == row.pos:
                enhanced_size = "-"  # Reset to avoid showing wrong size data
            
            # Enhanced anchor extraction: first look in object ("_anchorPoint"), else in packed transform extension
            anc = "-"
            
            # First try: Look in object block for "_anchorPoint"
            if isinstance(obj.get("_anchorPoint"), dict):
                anc = f"({_num(obj['_anchorPoint'].get('x', 0))},{_num(obj['_anchorPoint'].get('y', 0))})"
            elif isinstance(obj.get("_anchorPoint"), list) and len(obj["_anchorPoint"]) >= 2:
                anc = f"({_num(obj['_anchorPoint'][0])},{_num(obj['_anchorPoint'][1])})"
            
            # Second try: Look in packed transform extension (some versions embed it right after scale)
            elif anc == "-":
                # Try to find transform data in the raw row for anchor extraction
                trs_data = None
                for idx in [IDX_TRS, IDX_TRS-1, IDX_TRS+1, -1]:  # Last item sometimes contains transforms
                    if (idx == -1 and len(raw) > 0) or (0 <= idx < len(raw)):
                        candidate = raw[idx] if idx != -1 else raw[-1]
                        if isinstance(candidate, list) and len(candidate) >= 9:
                            # Check if this looks like transform data (numbers)
                            if all(isinstance(x, (int, float)) for x in candidate[:9]):
                                trs_data = candidate
                                break
                
                if trs_data and len(trs_data) > 9:
                    # Check if there are anchor point values after the standard 9 transform components
                    # Format: [px,py,pz, rx,ry,rz, sx,sy,sz, anchor_x, anchor_y, ...]
                    if len(trs_data) >= 11:  # At least 9 transform + 2 anchor components
                        anchor_x, anchor_y = trs_data[9], trs_data[10]
                        # Only show if not default (0.5, 0.5)
                        if anchor_x != 0.5 or anchor_y != 0.5:
                            anc = f"({_num(anchor_x)},{_num(anchor_y)})"
                        else:
                            anc = "(0.5,0.5)"  # Show default anchor if present
            # Use enhanced size if found in object block
            final_size = enhanced_size if enhanced_size != "-" else row.size
            
            n=Node(
                name=row.name or f"node_{block_id}_{i}",
                cls =class_of(templates[row.tpl][0]),
                key =(block_id,i),
                pos=row.pos, rot=row.rot, scale=row.scale, size=final_size, anc=anc,
                comps=row.components
            )
            self._collect_refs(raw,assets,n)
            # Also collect refs from the object data
            self._collect_refs(obj,assets,n)
            nodes[n.key]=n

        # Unified child/parent graph construction
        # 1. Prefer _children array from object JSON
        # 2. Fall back to parent index in packed row
        # 3. Deduplicate links
        child_keys = set()
        
        # First pass: Process _children arrays from object JSON (preferred)
        for i, raw in enumerate(rows):
            g_idx = offs[block_id] + i
            obj = data[g_idx] if 0 <= g_idx < len(data) and isinstance(data[g_idx], dict) else {}
            
            if "_children" in obj and isinstance(obj["_children"], list):
                parent_node = nodes[(block_id, i)]
                for child_ref in obj["_children"]:
                    if isinstance(child_ref, int):
                        child_key = (block_id, child_ref)
                        if child_key in nodes and child_key not in child_keys:
                            parent_node.add_child(nodes[child_key])
                            child_keys.add(child_key)
        
        # Second pass: Fall back to parent index in packed row for nodes without established relationships
        for i, raw in enumerate(rows):
            child_key = (block_id, i)
            if child_key not in child_keys:  # Only process if not already linked
                p = raw[2] if len(raw) > 2 and isinstance(raw[2], int) else None
                if p is not None and (block_id, p) in nodes:
                    nodes[(block_id, p)].add_child(nodes[child_key])
                    child_keys.add(child_key)
        
        # Third pass: Process other child lists from packed data for remaining unlinked nodes
        for i, raw in enumerate(rows):
            parent_node = nodes[(block_id, i)]
            for lst in raw[3:]:
                if isinstance(lst, list) and lst and all(isinstance(x, (int, float)) for x in lst):
                    for off in map(int, lst):
                        child_key = (block_id, i + (-off) if off < 0 else off)
                        if (child_key in nodes and 
                            child_key not in child_keys and 
                            child_key != parent_node.key):
                            parent_node.add_child(nodes[child_key])
                            child_keys.add(child_key)
                    break
        self._p_h(nodes); self._p_assets(assets, nodes); self._p_refs(nodes)
    
    def _extract_sprite_frames_and_assets(self, data: List[Any]) -> None:
        """Extract sprite frame details and asset information from pack data."""
        if len(data) < self.IDX_FIRST:
            return
        
        # Initialize registries
        if not hasattr(self, 'sprite_frames'):
            self.sprite_frames = {}
        if not hasattr(self, 'asset_uuid_map'):
            self.asset_uuid_map = {}
        
        # Search recursively through all data blocks for sprite frames and assets
        def search_for_assets(obj, depth=0):
            if depth > 10:  # Prevent infinite recursion
                return
                
            if isinstance(obj, dict):
                # Check for sprite frame data
                if "name" in obj and "rect" in obj and "originalSize" in obj:
                    frame_name = obj["name"]
                    rect = obj.get("rect", [0, 0, 0, 0])
                    original_size = obj.get("originalSize", [0, 0])
                    offset = obj.get("offset", [0, 0])
                    cap_insets = obj.get("capInsets", [0, 0, 0, 0])
                    
                    self.sprite_frames[frame_name] = {
                        "name": frame_name,
                        "rect": rect,
                        "originalSize": original_size,
                        "offset": offset,
                        "capInsets": cap_insets,
                        "width": original_size[0] if len(original_size) >= 1 else 0,
                        "height": original_size[1] if len(original_size) >= 2 else 0
                    }
                    
                    print(f"    Found sprite frame: '{frame_name}' ({original_size[0]}×{original_size[1]}) rect={rect}")
                
                # Check for UUID references
                if "__uuid__" in obj:
                    uuid = obj["__uuid__"]
                    if "name" in obj:
                        self.asset_uuid_map[uuid] = obj["name"]
                    
                # Recursively search dictionary values
                for value in obj.values():
                    search_for_assets(value, depth + 1)
                    
            elif isinstance(obj, list):
                # Check if this is a sprite frame list
                for item in obj:
                    if isinstance(item, dict) and "name" in item and "rect" in item:
                        frame_name = item["name"]
                        rect = item.get("rect", [0, 0, 0, 0])
                        original_size = item.get("originalSize", [0, 0])
                        offset = item.get("offset", [0, 0])
                        cap_insets = item.get("capInsets", [0, 0, 0, 0])
                        
                        self.sprite_frames[frame_name] = {
                            "name": frame_name,
                            "rect": rect,
                            "originalSize": original_size,
                            "offset": offset,
                            "capInsets": cap_insets,
                            "width": original_size[0] if len(original_size) >= 1 else 0,
                            "height": original_size[1] if len(original_size) >= 2 else 0
                        }
                        
                        print(f"    Found sprite frame: '{frame_name}' ({original_size[0]}×{original_size[1]}) rect={rect}")
                    else:
                        search_for_assets(item, depth + 1)
        
        # Search all data blocks
        for block in data[self.IDX_FIRST:]:
            search_for_assets(block)
    
    def _extract_embedded_animation_clips(self, data: List[Any]) -> None:
        """Extract embedded animation clips from the first data block and add to asset registry."""
        if len(data) < self.IDX_FIRST:
            return
        
        # Look for animation clip data in the first blocks 
        first_blocks = data[self.IDX_FIRST:self.IDX_FIRST + 3]  # Check first few blocks
        
        def search_recursively(obj, depth=0):
            """Recursively search for animation clip data in nested structures."""
            if depth > 5:  # Prevent infinite recursion
                return
                
            if isinstance(obj, list):
                # Check if this list is an animation clip
                if (len(obj) >= 6 and 
                    isinstance(obj[1], str) and 
                    isinstance(obj[2], (int, float)) and 
                    isinstance(obj[5], dict)):
                    self._check_animation_clip_item(obj)
                
                # Recursively search nested lists
                for item in obj:
                    search_recursively(item, depth + 1)
            elif isinstance(obj, dict):
                # Search dictionary values
                for value in obj.values():
                    search_recursively(value, depth + 1)
        
        # Search all blocks recursively
        for block_idx, block in enumerate(first_blocks):
            search_recursively(block)
    
    def _check_animation_clip_item(self, item: List[Any]) -> None:
        """Check if an item is an animation clip and extract its data."""
        # Check if this looks like animation clip data: [tpl_idx, name, duration, wrapMode, sample, curveData]
        if (len(item) >= 6 and 
            isinstance(item[1], str) and  # name
            isinstance(item[2], (int, float)) and  # duration
            isinstance(item[5], dict)):  # curveData
            
            clip_name = item[1]
            duration = float(item[2])
            wrap_mode = item[3]
            sample_rate = item[4]
            curves_data = item[5]
            
            print(f"    Found embedded animation clip: '{clip_name}' ({duration}s, wrap={wrap_mode}, sample={sample_rate})")
            
            # Parse and display curve information
            if "paths" in curves_data:
                paths = curves_data["paths"]
                for node_name, node_curves in paths.items():
                    if isinstance(node_curves, dict) and "props" in node_curves:
                        props = node_curves["props"]
                        for prop_name, keyframes in props.items():
                            if isinstance(keyframes, list) and keyframes:
                                print(f"      - {node_name}.{prop_name}: {len(keyframes)} keyframes")
                                
                                # Show detailed keyframe information
                                for i, kf in enumerate(keyframes[:4]):  # Show first 4 keyframes
                                    if isinstance(kf, dict) and "frame" in kf and "value" in kf:
                                        frame_time = kf["frame"]
                                        value = kf["value"]
                                        
                                        # Format value based on type
                                        if isinstance(value, list) and len(value) == 3:
                                            # Position [x, y, z]
                                            value_str = f"pos({value[0]:.1f}, {value[1]:.1f}, {value[2]:.1f})"
                                        elif isinstance(value, (int, float)):
                                            # Rotation angle
                                            value_str = f"{value}°"
                                        else:
                                            value_str = str(value)
                                        
                                        print(f"        [{i}] t={frame_time}s → {value_str}")
                                
                                # Show summary for longer sequences
                                if len(keyframes) > 4:
                                    print(f"        ... ({len(keyframes) - 4} more keyframes)")
            
            # Store in asset registry for component decoders to use
            if not hasattr(self, 'embedded_animations'):
                self.embedded_animations = {}
            self.embedded_animations[clip_name] = {
                "name": clip_name,
                "duration": duration,
                "wrap_mode": wrap_mode,
                "sample_rate": sample_rate,
                "curves": curves_data,
                "class": "cc.AnimationClip"
            }

    @staticmethod
    def _blocks(q:list)->List[list]:
        out,ql=[],list(q)
        while ql:
            b=ql.pop(0)
            if isinstance(b,list):
                if b and isinstance(b[0],list) and isinstance(b[0][0],int): out.append(b)
                elif b and isinstance(b[0],list): ql[0:0]=b
        return out
    def _block_offs(self,blks):
        offs,cur=[],self.IDX_FIRST
        for b in blks: offs.append(cur); cur+=len(b)
        return offs
    def _scene_block(self,blks,tpls,class_of):
        for i in reversed(range(len(blks))):
            for row in blks[i]:
                if isinstance(row,list) and row and isinstance(row[0],int):
                    if class_of(tpls[row[0]][0])=="cc.Scene": return i
        return len(blks)-1

    def _asset_index(self,templates,class_of):
        idx={}
        assets={} # Store assets with synthetic paths for component decoders
        
        for s,p in self.paths.items():
            ti=int(s)
            if 0<=ti<len(templates) and isinstance(templates[ti],list) and isinstance(templates[ti][0],int):
                tpl=templates[ti]
                cls_name = class_of(tpl[0])
                asset_name = tpl[1] if len(tpl)>1 and isinstance(tpl[1],str) else None
                
                idx[ti]=AssetInfo(ti, cls_name,
                                  asset_name,
                                  p[0] if isinstance(p,list) and p else "(no path)")
                
                # Store assets with readable synthetic paths for known asset classes
                if cls_name in ["cc.SpriteFrame", "cc.Material", "cc.Font", "cc.Texture2D", "cc.AudioClip", "cc.Prefab"]:
                    # Create synthetic path: assets/<cls>/<name or tplIdx>
                    if asset_name:
                        synthetic_path = f"assets/{cls_name}/{asset_name}"
                    else:
                        synthetic_path = f"assets/{cls_name}/template_{ti}"
                    
                    assets[synthetic_path] = {
                        "template_index": ti,
                        "class": cls_name,
                        "name": asset_name,
                        "path": p[0] if isinstance(p,list) and p else "(no path)",
                        "synthetic_path": synthetic_path
                    }
        
        # Store assets registry for component decoders to use
        self.assets = assets
        return idx
    @staticmethod
    def _collect_refs(itm,assets,n):
        if isinstance(itm,bool): return
        if isinstance(itm,int) and itm in assets: n.assets.add(itm)
        elif isinstance(itm,list):
            for sub in itm: CocosBundle._collect_refs(sub,assets,n)
        elif isinstance(itm,dict):
            for sub in itm.values(): CocosBundle._collect_refs(sub,assets,n)

    @staticmethod
    def _p_h(nodes):
        """Recursive pretty-print with indentation, guaranteeing single root (scene)."""
        print("    Scene graph:")
        all_nodes = list(nodes.values())
        if not all_nodes: 
            print("      [no nodes]")
            return
            
        # Find all children to identify root candidates
        children_keys = set()
        for node in all_nodes:
            for child in node.children:
                children_keys.add(child.key)
        
        # Find the root - prefer Scene node, otherwise any node without a parent
        scene_nodes = [n for n in all_nodes if n.cls == "cc.Scene"]
        root_candidates = [n for n in all_nodes if n.key not in children_keys]
        
        # Guarantee a single root
        if scene_nodes:
            # Prefer Scene node as root
            root = scene_nodes[0]
        elif root_candidates:
            # Use first root candidate if no Scene node
            root = root_candidates[0]
        else:
            # Fallback: use first node if all seem to be children (circular or orphaned)
            root = all_nodes[0]
        
        # Recursive pretty-print starting from root
        def print_node(node: Node, indent: int = 0):
            """Recursively print node and its children with proper indentation."""
            prefix = "  " * indent + "- "
            
            # Enhanced node line format: name (cls) pos=… rot=… scale=… anc=… size=…
            transform_parts = []
            if node.pos != "-":
                transform_parts.append(f"pos={node.pos}")
            if node.rot != "-":
                transform_parts.append(f"rot={node.rot}")
            if node.scale != "-":
                transform_parts.append(f"scale={node.scale}")
            if node.anc != "-":
                transform_parts.append(f"anc={node.anc}")
            if node.size != "-":
                transform_parts.append(f"size={node.size}")
            
            transform_str = f" {' '.join(transform_parts)}" if transform_parts else ""
            
            # Main node line
            print(f"      {prefix}{node.name} ({node.cls}){transform_str}")
            
            # Under each node, list decoded components with their __str__
            if node.comps:
                for comp in node.comps:
                    comp_indent = "  " * (indent + 1) + "    "  # Extra indentation for components
                    if comp.decoded_component:
                        # Use the decoded component's __str__ method
                        decoded_str = str(comp.decoded_component)
                        print(f"      {comp_indent}└─ {decoded_str}")
                    else:
                        # Fallback to basic component info
                        print(f"      {comp_indent}└─ {comp}")
            
            # Show asset references if any
            if node.assets:
                asset_indent = "  " * (indent + 1) + "    "
                print(f"      {asset_indent}[assets: {','.join(map(str,sorted(node.assets)))}]")
            
            # Recursively print children with increased indentation
            for child in sorted(node.children, key=lambda c: c.name):
                print_node(child, indent + 1)
        
        # Start recursive printing from root
        print_node(root)
        
        # Print any orphaned nodes that aren't connected to the main tree
        visited = set()
        def mark_visited(node: Node):
            if node.key in visited:
                return
            visited.add(node.key)
            for child in node.children:
                mark_visited(child)
        
        mark_visited(root)
        
        orphaned = [n for n in all_nodes if n.key not in visited]
        if orphaned:
            print("\n      Orphaned nodes (not connected to main tree):")
            for node in orphaned:
                print_node(node)
    def _p_assets(self, assets_dict, nodes):
        """Print asset table - full table if --assets flag is set, otherwise only referenced assets."""
        if not assets_dict:
            return
        
        if self.show_full_assets:
            # Show full asset table
            print("    File assets (full table):")
            for k in sorted(assets_dict):
                print("      " + assets_dict[k].row())
        else:
            # Show only referenced assets
            referenced_assets = set()
            for node in nodes.values():
                referenced_assets.update(node.assets)
            
            if referenced_assets:
                print("    File assets (referenced only):")
                for asset_id in sorted(referenced_assets):
                    if asset_id in assets_dict:
                        print("      " + assets_dict[asset_id].row())
            else:
                print("    File assets: [none referenced]")
    def _p_refs(self, nodes):
        """Enhanced asset reference display with comprehensive sprite frame name resolution and animation targets."""
        print("    Node → asset refs:")
        anyr = False
        
        # Build comprehensive asset reference mapping
        for n in sorted(nodes.values(), key=lambda x: x.name):
            asset_refs = []
            
            # Process components for asset references
            if hasattr(n, 'comps') and n.comps:
                for comp in n.comps:
                    if comp.decoded_component:
                        # Sprite frame references - enhanced to use sprite_frames registry
                        if isinstance(comp.decoded_component, SpriteComponent):
                            if hasattr(comp.decoded_component, '_resolved_frame_name') and comp.decoded_component._resolved_frame_name:
                                frame_name = comp.decoded_component._resolved_frame_name
                                frame_data = getattr(comp.decoded_component, '_resolved_frame_data', {})
                                width = frame_data.get('width', 0)
                                height = frame_data.get('height', 0)
                                asset_refs.append(f'SpriteFrame[{frame_name}] ({width}×{height})')
                            elif comp.decoded_component.sprite_frame:
                                # Fallback: try to resolve using sprite_frames registry
                                if hasattr(self, 'sprite_frames') and self.sprite_frames:
                                    # Try direct node name match
                                    if n.name in self.sprite_frames:
                                        frame_data = self.sprite_frames[n.name]
                                        width = frame_data.get('width', 0)
                                        height = frame_data.get('height', 0)
                                        asset_refs.append(f'SpriteFrame[{n.name}] ({width}×{height})')
                                    else:
                                        # Smart matching for common patterns
                                        matched_frame = None
                                        for frame_name in self.sprite_frames.keys():
                                            if (frame_name.lower() in n.name.lower() or 
                                                n.name.lower() in frame_name.lower()):
                                                matched_frame = frame_name
                                                break
                                        
                                        if matched_frame:
                                            frame_data = self.sprite_frames[matched_frame]
                                            width = frame_data.get('width', 0)
                                            height = frame_data.get('height', 0)
                                            asset_refs.append(f'SpriteFrame[{matched_frame}] ({width}×{height})')
                                        else:
                                            asset_refs.append(f"SpriteFrame[#{comp.decoded_component.sprite_frame}]")
                                else:
                                    asset_refs.append(f"SpriteFrame[#{comp.decoded_component.sprite_frame}]")
                        
                        # Font references for labels
                        elif isinstance(comp.decoded_component, LabelComponent):
                            if comp.decoded_component.font_asset:
                                asset_refs.append(f"Font[asset_id={comp.decoded_component.font_asset}]")
                            else:
                                asset_refs.append("Font[default]")
                        
                        # Animation clip references with targets - enhanced to use embedded_animations
                        elif isinstance(comp.decoded_component, AnimationComponent):
                            if comp.decoded_component.clips:
                                for clip in comp.decoded_component.clips:
                                    targets = list(clip.curves.keys()) if hasattr(clip, 'curves') and clip.curves else []
                                    if targets:
                                        asset_refs.append(f"AnimationClip['{clip.name}'] → targets: [{', '.join(targets)}]")
                                    else:
                                        asset_refs.append(f"AnimationClip['{clip.name}']")
                            else:
                                # Fallback: check for embedded animations
                                if hasattr(self, 'embedded_animations') and self.embedded_animations:
                                    clip_names = list(self.embedded_animations.keys())
                                    if clip_names:
                                        # Extract targets from the first available clip
                                        clip_data = list(self.embedded_animations.values())[0]
                                        targets = []
                                        if 'curves' in clip_data and 'paths' in clip_data['curves']:
                                            targets = list(clip_data['curves']['paths'].keys())
                                        
                                        if targets:
                                            asset_refs.append(f"AnimationClip[{clip_names[0]}] → targets: [{', '.join(targets)}]")
                                        else:
                                            asset_refs.append(f"AnimationClip[{clip_names[0]}]")
            
            # Collect traditional asset references if not already covered
            if n.assets:
                for asset_id in sorted(n.assets):
                    if not any(str(asset_id) in ref for ref in asset_refs):
                        asset_refs.append(f"Asset[id={asset_id}]")
            
            if asset_refs:
                anyr = True
                print(f"      {n.name:28} → {', '.join(asset_refs)}")
        
        if not anyr: 
            print("      [none]")

def test_component_decoder():
    """Test the ComponentDecoder engine functionality."""
    print("Testing ComponentDecoder Engine...")
    
    # Create engine
    engine = ComponentDecoderEngine()
    
    # Test asset registry
    asset_registry = {
        100: {"class": "cc.SpriteFrame", "name": "button_bg.png"},
        101: {"class": "cc.Font", "name": "arial.ttf"},
    }
    
    # Test Label component
    print("\n1. Testing cc.Label decoder:")
    helper = DecodeHelper(
        asset_registry=asset_registry,
        property_names=["_N$verticalAlign", "_N$horizontalAlign", "_string", "_fontSize", "_isSystemFontUsed", "_N$cacheMode", "_lineHeight", "_styleFlags", "_N$overflow", "_enableWrapText", "node", "_materials", "_N$file"]
    )
    
    # Example Label component data: [template_index, vertAlign, horizAlign, text, fontSize, systemFont, cacheMode, lineHeight, styleFlags, overflow, wrapText, node, materials, font_file]
    label_data = [5, 1, 0, "Hello World", 24, True, 0, 36, 0, 0, True, None, [], 101]
    
    decoded_label = engine.decode_component("cc.Label", label_data, ["_N$verticalAlign", "_N$horizontalAlign", "_string", "_fontSize", "_isSystemFontUsed", "_N$cacheMode", "_lineHeight", "_styleFlags", "_N$overflow", "_enableWrapText", "node", "_materials", "_N$file"], helper)
    
    print(f"   Decoded: {decoded_label.component_type}")
    print(f"   Text: '{decoded_label.text}'")
    print(f"   Font size: {decoded_label.font_size}")
    print(f"   Font asset: {decoded_label.font_asset}")
    print(f"   Asset refs: {decoded_label.asset_refs}")
    print(f"   Properties count: {len(decoded_label.properties)}")
    
    # Test Sprite component
    print("\n2. Testing cc.Sprite decoder:")
    # Example Sprite component data: [template_index, sizeMode, type, trimmed, enabled, blendFactor, fillRange, node, materials, spriteFrame]
    sprite_data = [10, 0, 1, True, True, 770, 1.0, None, [], 100]
    
    decoded_sprite = engine.decode_component("cc.Sprite", sprite_data, ["_sizeMode", "_type", "_isTrimmedMode", "_enabled", "_dstBlendFactor", "_fillRange", "node", "_materials", "_spriteFrame"], helper)
    
    print(f"   Decoded: {decoded_sprite.component_type}")
    print(f"   Size mode: {decoded_sprite.size_mode}")
    print(f"   Sprite type: {decoded_sprite.sprite_type}")
    print(f"   Sprite frame: {decoded_sprite.sprite_frame}")
    print(f"   Asset refs: {decoded_sprite.asset_refs}")
    print(f"   Properties count: {len(decoded_sprite.properties)}")
    
    # Test unknown component (fallback decoder)
    print("\n3. Testing unknown component (fallback decoder):")
    unknown_data = [15, "some_value", 42, [4, 255, 128, 64, 255], True]
    decoded_unknown = engine.decode_component("cc.UnknownComponent", unknown_data, ["prop1", "prop2", "prop3", "prop4"], helper)
    
    print(f"   Decoded: {decoded_unknown.component_type}")
    print(f"   Properties: {decoded_unknown.properties}")
    
    # Test registry functionality
    print("\n4. Testing decoder registry:")
    print(f"   Supported types: {engine.get_supported_types()}")
    
    # Test registering a new decoder
    class CustomDecoder(ComponentDecoder):
        @property
        def component_type(self) -> str:
            return "cc.Custom"
        
        def decode(self, raw_row, prop_names, helper) -> DecodedComponent:
            return DecodedComponent(
                component_type=self.component_type,
                template_index=raw_row[0] if raw_row else -1,
                properties={"custom": "test"}
            )
    
    engine.register_decoder("cc.Custom", CustomDecoder)
    print(f"   After registering custom decoder: {engine.get_supported_types()}")
    
    custom_component = engine.decode_component("cc.Custom", [99], [], helper)
    print(f"   Custom component: {custom_component.component_type}, properties: {custom_component.properties}")
    
    print("\n✓ ComponentDecoder engine test completed successfully!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Cocos Creator Pack Inspector")
    parser.add_argument("config", nargs="?", help="Bundle config JSON file")
    parser.add_argument("--test", action="store_true", help="Run ComponentDecoder engine test")
    parser.add_argument("--assets", action="store_true", help="Show full asset table (default: only referenced assets)")
    
    args = parser.parse_args()
    
    if args.test:
        test_component_decoder()
    elif not args.config:
        parser.print_help()
        sys.exit(1)
    else:
        bundle = CocosBundle(args.config)
        bundle.show_full_assets = args.assets
        bundle.run()
