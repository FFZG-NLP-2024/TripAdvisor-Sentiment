from transformers import pipeline

def check_model():
    try:
        # Load the model pipeline for text classification
        classifier = pipeline("text-classification", model="nhull/distilbert-sentiment-model")
        
        # Test the model with a sample input
        test_input = """Don't know what to think
The public spaces are dark, the elevators are dark; almost to the point of not being able to see the numbers on the buttons, but the worst for me was the poor lighting in the room itself. The bathroom is lit by an illuminated mirror only. It is nowhere near bright enough to put on makeup etc. I had to go out of the bathroom and open the blinds to try to get some natural light. Bathroom too small to use the blowdryer with the door closed, you have to leave the door open and kind of use the hallway space as well. But, the weirdest thing is the window into the bath from the bedroom. It is covered by a sheer net curtain, and you can see in quite clearly. Also, when you take a shower, there is a curtain on both sides of you and they both kind of blow into the tub making you feel like you are wrapped up like a mummy. 
The room was hot, we had to use the air conditioning the whole time (in February!) but it blows right from behind your head while in bed, making it kind of uncomfortable at times. One night, the heater/AC unit started banging in the middle of the night, and took quite a while to stop. Also, there are no black-out curtains, just a rather mangled venetian blind that lets in quite a lot of light, so if you need a dark room for sleeping you're out of luck here.
The room was nice and clean, the bed & pillows were comfortable, the toiletries were really nice, we didn't have a problem with noise from other rooms or hallway. But the size is quite comical, even considering you don't spend much time in your room. You have to take turns moving around the room, being careful not to bang your head on the wall-mounted TV.
Location is also good, near lots of cafes and less than a block to either Central Park, or the big Columbus Circle subway station.
Would probably not want to stay here again."""

        result = classifier(test_input)
        
        # Display the result
        print("Model works! Here's the prediction:")
        print(result)

    except Exception as e:
        print("An error occurred while checking the model:")
        print(e)

if __name__ == "__main__":
    check_model()
