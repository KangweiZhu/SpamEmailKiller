import sys
import re
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n')

def parse_eml(eml_path):
    with open(eml_path, 'rb') as eml_file:
        msg = BytesParser(policy=policy.default).parse(eml_file)

    sender = msg.get('From', '(No From)')
    recipient = msg.get('To', '(No To)')
    subject = msg.get('Subject', '(No Subject)')
    date = msg.get('Date', '(No Date)')
    body_parts = []
    attachments = []
    links = []

    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = part.get_content_disposition()
        payload = part.get_payload(decode=True)

        if payload:
            charset = part.get_content_charset('utf-8')
            decoded = payload.decode(charset, errors='ignore')

            if content_type == 'text/plain':
                body_parts.append(decoded)
            elif content_type == 'text/html':
                body_parts.append(extract_text_from_html(decoded))
        
        if content_disposition == 'attachment':
            attachments.append({
                'filename': part.get_filename(),
                'content_type': content_type
            })

    full_body = "\n\n".join(body_parts).strip()
    links = re.findall(r'https?://[^\s\'"<>]+', full_body)

    return {
        'sender': sender,
        'recipient': recipient,
        'subject': subject,
        'date': date,
        'body': full_body,
        'attachments': attachments,
        'links': links
    }

def save_email_as_txt(parsed, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"From: {parsed['sender']}\n")
        f.write(f"To: {parsed['recipient']}\n")
        f.write(f"Subject: {parsed['subject']}\n")
        f.write(f"Date: {parsed['date']}\n")
        f.write("\n--- Body ---\n\n")
        f.write(parsed['body'] + "\n\n")

        if parsed['links']:
            f.write("--- Links ---\n")
            for link in parsed['links']:
                f.write(link + "\n")
            f.write("\n")

        if parsed['attachments']:
            f.write("--- Attachments ---\n")
            for att in parsed['attachments']:
                f.write(f"{att['filename']} ({att['content_type']})\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process.py input.eml output.txt")
        sys.exit(1)
    
    eml_file = sys.argv[1]
    txt_file = sys.argv[2]
    parsed_email = parse_eml(eml_file)
    save_email_as_txt(parsed_email, txt_file)
    print(f"Saved: {txt_file}")