// Framework-specific optimizations
const app = {
    state: {
        patterns: null,
        currentView: 'both',
        selectedPatterns: [],
        zoomLevel: 1,
        rotation: 0,
        scene: null,
        camera: null,
        renderer: null,
        model: null
    },
    
    async init() {
        await this.loadData();
        await this.setupThreeJS();
        await this.setupTensorFlow();
        this.setupEventListeners();
        this.render();
    },
    
    async loadData() {
        const response = await fetch('pattern_explorer_data.json');
        const data = await response.json();
        this.state.patterns = data.patterns;
        this.state.mandalas = data.mandalas;
        this.state.concretions = data.concretions;
    },
    
    async setupThreeJS() {
        // Initialize Three.js scene
        this.state.scene = new THREE.Scene();
        this.state.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.state.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.state.renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('app').appendChild(this.state.renderer.domElement);
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.state.scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 0);
        this.state.scene.add(directionalLight);
        
        // Position camera
        this.state.camera.position.z = 5;
        
        // Create pattern geometries
        this.createPatternGeometries();
    },
    
    async setupTensorFlow() {
        // Load TensorFlow.js model for pattern analysis
        this.state.model = await tf.loadLayersModel('pattern_analysis_model/model.json');
    },
    
    createPatternGeometries() {
        this.state.patterns.forEach((pattern, index) => {
            const geometry = new THREE.BufferGeometry();
            const vertices = this.generatePatternVertices(pattern);
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            
            const material = new THREE.MeshPhongMaterial({
                color: this.getPatternColor(pattern),
                wireframe: true,
                transparent: true,
                opacity: 0.8
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData = { patternIndex: index };
            this.state.scene.add(mesh);
        });
    },
    
    generatePatternVertices(pattern) {
        const vertices = [];
        const lattice = pattern.lattice;
        
        for (let i = 0; i < lattice.length; i++) {
            for (let j = 0; j < lattice[i].length; j++) {
                if (lattice[i][j]) {
                    // Create E8 lattice-inspired vertex positions
                    const x = (i - lattice.length/2) * 0.5;
                    const y = (j - lattice[i].length/2) * 0.5;
                    const z = pattern.stats.depth * 2 - 1;
                    vertices.push(x, y, z);
                }
            }
        }
        
        return vertices;
    },
    
    getPatternColor(pattern) {
        const { depth, complexity, harmony } = pattern.stats;
        // Create color based on pattern features
        return new THREE.Color(
            depth,
            complexity,
            harmony
        );
    },
    
    setupEventListeners() {
        // Zoom controls
        document.querySelector('.zoom-controls button:first-child').addEventListener('click', () => this.zoomOut());
        document.querySelector('.zoom-controls button:last-child').addEventListener('click', () => this.zoomIn());
        
        // Rotation controls
        document.querySelector('.rotation-controls button:first-child').addEventListener('click', () => this.rotateLeft());
        document.querySelector('.rotation-controls button:last-child').addEventListener('click', () => this.rotateRight());
        
        // View controls
        document.querySelectorAll('.control-group button').forEach(button => {
            button.addEventListener('click', () => this.toggleView(button.textContent.toLowerCase()));
        });
        
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
    },
    
    render() {
        requestAnimationFrame(() => this.render());
        
        // Update Three.js scene
        this.state.scene.rotation.y += 0.005;
        this.state.renderer.render(this.state.scene, this.state.camera);
        
        // Update pattern cards
        this.updatePatternCards();
    },
    
    updatePatternCards() {
        const container = document.querySelector('.container');
        container.innerHTML = '';
        
        this.state.patterns.forEach((pattern, index) => {
            const card = this.createPatternCard(pattern, index);
            container.appendChild(card);
        });
    },
    
    createPatternCard(pattern, index) {
        const card = document.createElement('div');
        card.className = 'pattern-card';
        card.innerHTML = `
            <div class="pattern-content" style="--zoom-level: ${this.state.zoomLevel}">
                ${this.state.currentView === 'both' ? 
                    this.state.mandalas[index] + '\\n\\n' + this.state.concretions[index] :
                    this.state.currentView === 'mandala' ? this.state.mandalas[index] : this.state.concretions[index]}
            </div>
            <div class="pattern-info">
                <span>Pattern ${index + 1}</span>
                <div class="pattern-stats">
                    Depth: ${pattern.stats.depth.toFixed(2)} | 
                    Complexity: ${pattern.stats.complexity.toFixed(2)} | 
                    Harmony: ${pattern.stats.harmony.toFixed(2)}
                </div>
            </div>
            <div class="lattice-grid" style="display: none;">
                ${this.generateLatticeHTML(pattern.lattice)}
            </div>
        `;
        
        card.addEventListener('click', (e) => {
            if (e.target.closest('.lattice-grid')) return;
            this.handleCardClick(card, index);
        });
        
        return card;
    },
    
    generateLatticeHTML(lattice) {
        return lattice.map(row => 
            `<div class="lattice-row">${
                row.map(cell => 
                    `<div class="lattice-cell ${cell ? 'active' : ''}"></div>`
                ).join('')
            }</div>`
        ).join('');
    },
    
    handleCardClick(card, index) {
        card.classList.toggle('expanded');
        if (card.classList.contains('expanded')) {
            this.state.selectedPatterns.push(index);
            if (this.state.selectedPatterns.length === 2) {
                this.showComparison(this.state.selectedPatterns[0], this.state.selectedPatterns[1]);
            }
        } else {
            this.state.selectedPatterns = this.state.selectedPatterns.filter(i => i !== index);
        }
    },
    
    showComparison(index1, index2) {
        const view = document.getElementById('comparisonView');
        const card1 = document.getElementById('comparisonCard1');
        const card2 = document.getElementById('comparisonCard2');
        
        card1.innerHTML = `
            <div class="pattern-content">${this.state.mandalas[index1]}</div>
            <div class="pattern-info">Pattern ${index1 + 1}</div>
        `;
        
        card2.innerHTML = `
            <div class="pattern-content">${this.state.mandalas[index2]}</div>
            <div class="pattern-info">Pattern ${index2 + 1}</div>
        `;
        
        view.classList.add('active');
    },
    
    zoomIn() {
        this.state.zoomLevel = Math.min(this.state.zoomLevel + 0.1, 2);
        this.updateZoom();
    },
    
    zoomOut() {
        this.state.zoomLevel = Math.max(this.state.zoomLevel - 0.1, 0.5);
        this.updateZoom();
    },
    
    updateZoom() {
        document.querySelectorAll('.pattern-content').forEach(content => {
            content.style.setProperty('--zoom-level', this.state.zoomLevel);
        });
    },
    
    rotateLeft() {
        this.state.rotation -= 15;
        this.updateRotation();
    },
    
    rotateRight() {
        this.state.rotation += 15;
        this.updateRotation();
    },
    
    updateRotation() {
        document.querySelectorAll('.pattern-card').forEach(card => {
            card.style.transform = `rotate(${this.state.rotation}deg)`;
        });
    },
    
    handleResize() {
        this.state.camera.aspect = window.innerWidth / window.innerHeight;
        this.state.camera.updateProjectionMatrix();
        this.state.renderer.setSize(window.innerWidth, window.innerHeight);
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => app.init()); 