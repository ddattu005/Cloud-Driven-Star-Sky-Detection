from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import os
import math

app = Flask(__name__)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh

def normalize_coordinates(stars):
    brightest_star = stars[0]
    normalized_stars = [(star[0] - brightest_star[0], star[1] - brightest_star[1]) for star in stars]
    return normalized_stars

def calculate_distances(stars):
    distances = []
    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            distances.append(math.sqrt((stars[i][0] - stars[j][0]) ** 2 + (stars[i][1] - stars[j][1]) ** 2))
    return sorted(distances)

def match_constellation(normalized_stars, known_constellations):
    test_distances = calculate_distances(normalized_stars)
    best_match = None
    best_score = float('inf')
    
    for constellation_name, constellation_stars in known_constellations.items():
        constellation_distances = calculate_distances(constellation_stars)
        
        # Normalize distances
        min_len = min(len(test_distances), len(constellation_distances))
        test_distances = test_distances[:min_len]
        constellation_distances = constellation_distances[:min_len]
        
        # Calculate score based on distance differences
        score = sum(abs(td - cd) for td, cd in zip(test_distances, constellation_distances))
        
        if score < best_score:
            best_score = score
            best_match = constellation_name
    
    return best_match

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            thresh = preprocess_image(file_path)
            stars = detect_stars(thresh)
            brightest_stars = get_brightest_stars(stars, thresh)
            normalized_stars = normalize_coordinates(brightest_stars)
            
            # Placeholder for known constellations
            known_constellations = {
                "Orion": [(5.9195, 7.4071), (5.9195, 9.6472), (5.9195, 8.1589), (5.9195, 7.4071), (5.9195, 5.9195), (5.9195, 4.2436), (5.9195, 3.4054)],
                "Ursa Major": [(11.0621, 61.7508), (11.0307, 56.3824), (11.0621, 53.6948), (11.0621, 48.7414), (11.0621, 47.1565), (11.0621, 41.4994), (11.0621, 40.9556)],
                "Pisces": [(1.5333, 15.345), (1.5333, 14.345), (1.5333, 13.345), (1.5333, 12.345), (1.5333, 11.345), (1.5333, 10.345), (1.5333, 9.345)],
                "Leo": [(10.2833, 11.967), (10.2833, 12.967), (10.2833, 13.967), (10.2833, 14.967), (10.2833, 15.967), (10.2833, 16.967), (10.2833, 17.967)],
                # Add more constellations as needed
            }
            
            result = match_constellation(normalized_stars, known_constellations)
            if result:
                result = f"Constellation matched: {result}"
            else:
                result = "No matching constellation found."
            
            return render_template('result.html', result=result)
    return render_template('index.html')

def detect_stars(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stars = []
    for contour in contours:
        # Get the moments to calculate the centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            stars.append((cX, cY))
    
    return stars

def get_brightest_stars(stars, thresh):
    # Dummy implementation for getting brightest stars
    return stars  # Replace with actual logic to get brightest stars

if __name__ == '__main__':
    app.run(debug=True)