"""
Attribute ontology for road user classification.

Defines the 8 attribute categories, their possible values, natural language
descriptions, and prompt templates used by the VLM auto-labeling pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AttributeValue:
    label: str
    description: str
    vlm_keywords: List[str]  # keywords the VLM might use to signal this class


@dataclass
class Attribute:
    name: str
    display_name: str
    values: List[AttributeValue]
    applicable_to: List[str]  # vehicle | pedestrian | cyclist | all
    vlm_question: str


ONTOLOGY: Dict[str, Attribute] = {
    "mobility": Attribute(
        name="mobility",
        display_name="Mobility State",
        applicable_to=["all"],
        vlm_question="Is the road user moving, stationary, or moving slowly?",
        values=[
            AttributeValue(
                label="moving",
                description="The road user is in motion at normal speed",
                vlm_keywords=["moving", "traveling", "driving", "walking", "riding", "in motion"],
            ),
            AttributeValue(
                label="stationary",
                description="The road user is not moving, stopped",
                vlm_keywords=["stopped", "stationary", "parked", "standing still", "not moving"],
            ),
            AttributeValue(
                label="slow_moving",
                description="The road user is moving at a very slow speed",
                vlm_keywords=["slow", "crawling", "barely moving", "inching"],
            ),
        ],
    ),
    "orientation": Attribute(
        name="orientation",
        display_name="Orientation",
        applicable_to=["all"],
        vlm_question="What is the orientation of the road user relative to the camera? Are they facing toward the camera, facing away, or moving laterally?",
        values=[
            AttributeValue(
                label="facing_toward",
                description="The road user is facing toward the camera",
                vlm_keywords=["facing camera", "facing toward", "front view", "head-on"],
            ),
            AttributeValue(
                label="facing_away",
                description="The road user is facing away from the camera",
                vlm_keywords=["facing away", "back view", "rear view", "moving away"],
            ),
            AttributeValue(
                label="lateral",
                description="The road user is moving perpendicular to the camera view",
                vlm_keywords=["sideways", "lateral", "side view", "profile", "crossing"],
            ),
        ],
    ),
    "occlusion": Attribute(
        name="occlusion",
        display_name="Occlusion Level",
        applicable_to=["all"],
        vlm_question="How much of the road user is occluded or hidden by other objects?",
        values=[
            AttributeValue(
                label="none",
                description="The road user is fully visible with no occlusion",
                vlm_keywords=["fully visible", "no occlusion", "clear view", "unobstructed"],
            ),
            AttributeValue(
                label="partial",
                description="Part of the road user is hidden behind other objects",
                vlm_keywords=["partially hidden", "partially occluded", "partially visible", "obstructed"],
            ),
            AttributeValue(
                label="heavy",
                description="Most of the road user is hidden, only a small portion visible",
                vlm_keywords=["mostly hidden", "heavily occluded", "barely visible", "significantly blocked"],
            ),
        ],
    ),
    "lighting": Attribute(
        name="lighting",
        display_name="Lighting Condition",
        applicable_to=["all"],
        vlm_question="What are the lighting conditions affecting the road user in this image?",
        values=[
            AttributeValue(
                label="well_lit",
                description="The road user is clearly illuminated",
                vlm_keywords=["well lit", "bright", "daylight", "clear lighting", "good visibility"],
            ),
            AttributeValue(
                label="low_light",
                description="The road user is in dim or dark conditions",
                vlm_keywords=["dark", "dim", "night", "low light", "poorly lit", "shadowed"],
            ),
            AttributeValue(
                label="backlit",
                description="The road user is backlit, creating a silhouette effect",
                vlm_keywords=["backlit", "silhouette", "glare", "against bright background", "overexposed background"],
            ),
        ],
    ),
    "size": Attribute(
        name="size",
        display_name="Apparent Size",
        applicable_to=["all"],
        vlm_question="How large does the road user appear in the frame relative to the image size?",
        values=[
            AttributeValue(
                label="small",
                description="The road user occupies a small portion of the frame, likely far away",
                vlm_keywords=["small", "far away", "distant", "tiny", "far"],
            ),
            AttributeValue(
                label="medium",
                description="The road user occupies a moderate portion of the frame",
                vlm_keywords=["medium", "moderate size", "middle distance"],
            ),
            AttributeValue(
                label="large",
                description="The road user occupies a large portion of the frame, likely close",
                vlm_keywords=["large", "close", "nearby", "prominent", "fills frame"],
            ),
        ],
    ),
    "posture": Attribute(
        name="posture",
        display_name="Posture",
        applicable_to=["pedestrian", "cyclist"],
        vlm_question="What is the body posture of this person? Are they upright, leaning, or crouched?",
        values=[
            AttributeValue(
                label="upright",
                description="The person is standing or moving in an upright position",
                vlm_keywords=["upright", "standing", "erect", "straight"],
            ),
            AttributeValue(
                label="leaning",
                description="The person is leaning forward, backward, or to the side",
                vlm_keywords=["leaning", "tilted", "bent forward", "hunched"],
            ),
            AttributeValue(
                label="crouched",
                description="The person is in a crouched or low position",
                vlm_keywords=["crouched", "crouching", "squatting", "low stance"],
            ),
        ],
    ),
    "group": Attribute(
        name="group",
        display_name="Group Size",
        applicable_to=["all"],
        vlm_question="Is this road user traveling alone, in a pair, or as part of a larger group?",
        values=[
            AttributeValue(
                label="solo",
                description="The road user is alone",
                vlm_keywords=["alone", "solo", "single", "by themselves", "individual"],
            ),
            AttributeValue(
                label="pair",
                description="Two road users traveling together",
                vlm_keywords=["pair", "two", "couple", "together", "duo"],
            ),
            AttributeValue(
                label="group",
                description="Three or more road users together",
                vlm_keywords=["group", "multiple", "several", "crowd", "many"],
            ),
        ],
    ),
    "attention": Attribute(
        name="attention",
        display_name="Attention State",
        applicable_to=["pedestrian"],
        vlm_question="Is this pedestrian paying attention to traffic, distracted, or using a phone?",
        values=[
            AttributeValue(
                label="attentive",
                description="The pedestrian appears to be paying attention to their surroundings",
                vlm_keywords=["looking ahead", "attentive", "aware", "watching traffic", "alert"],
            ),
            AttributeValue(
                label="distracted",
                description="The pedestrian appears distracted or not paying attention",
                vlm_keywords=["distracted", "looking away", "not paying attention", "head turned"],
            ),
            AttributeValue(
                label="phone_use",
                description="The pedestrian is visibly using a phone or device",
                vlm_keywords=["phone", "device", "looking at screen", "texting", "smartphone"],
            ),
        ],
    ),
}

# maps integer class indices to labels for each attribute
LABEL_MAPS: Dict[str, Dict[int, str]] = {
    attr_name: {i: v.label for i, v in enumerate(attr.values)}
    for attr_name, attr in ONTOLOGY.items()
}

# reverse maps from label string to index
LABEL_TO_IDX: Dict[str, Dict[str, int]] = {
    attr_name: {v.label: i for i, v in enumerate(attr.values)}
    for attr_name, attr in ONTOLOGY.items()
}

ROAD_USER_TYPES = ["vehicle", "pedestrian", "cyclist"]


def get_applicable_attributes(road_user_type: str) -> List[str]:
    """Returns attribute names that apply to a given road user type."""
    applicable = []
    for name, attr in ONTOLOGY.items():
        if "all" in attr.applicable_to or road_user_type in attr.applicable_to:
            applicable.append(name)
    return applicable


def get_num_classes(attribute_name: str) -> int:
    return len(ONTOLOGY[attribute_name].values)
