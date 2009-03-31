class GEMINI_NORTH(DataClassification):
    name = "GEMINI_NORTH"
    phuReqs = {
        "OBSERVAT":"Gemini-North",
        "TELESCOP":"Gemini-North"
        }
newtypes.append(GEMINI_NORTH())


class GEMINI_SOUTH(DataClassification):
    name = "GEMINI_SOUTH"
    phuReqs = {
        "OBSERVAT":"Gemini-South",
        "TELESCOP":"Gemini-South"
        }
newtypes.append(GEMINI_SOUTH())
