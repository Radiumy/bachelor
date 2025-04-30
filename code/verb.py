import spacy

nlp = spacy.load("en_core_web_sm")

def pos_tag_and_check_verbs(sentence):
    doc = nlp(sentence)
    has_verb = False
    
    print(f"{'Token':15} {'POS':6} {'TAG':6}")
    print("-" * 30)
    for token in doc:
        print(f"{token.text:15} {token.pos_:6} {token.tag_:6}")
        if token.pos_ == "VERB":
            has_verb = True
    
    return has_verb

user_input = input("Enter a sentence: ")
has_verb = pos_tag_and_check_verbs(user_input)
print(has_verb)