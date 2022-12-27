<html lang="en">
<head>
    <title>WebGPU Instance</title>
    <meta charset="utf-8">
</head>
<body>
<script type="module">
import * as THREE from '../build/three.module.js';

import WebGPURenderer from './jsm/renderers/webgpu/WebGPURenderer.js';
import WebGPU from './jsm/renderers/webgpu/WebGPU.js';
import { 
    Matrix3Uniform,
    Matrix4Uniform
} from './jsm/renderers/webgpu/WebGPUUniform.js';
import WebGPUStorageBuffer from './jsm/renderers/webgpu/WebGPUStorageBuffer.js';
import WebGPUUniformsGroup from './jsm/renderers/webgpu/WebGPUUniformsGroup.js';

let camera, scene, renderer;
let pointer;

// Math.PI
const MATH_PI = Math.PI;

const computeParams_density_pressure = [];
const computeParams_force = [];
const computeParams_integrate = [];

init().then( animate ).catch( error );

async function init() {
    if ( WebGPU.isAvailable() === false ) {
        document.body.appendChild( WebGPU.getErrorMessage() );
        throw 'No WebGPU support';
    }

    camera = new THREE.PerspectiveCamera( 70, window.innerWidth / window.innerHeight, 0.1, 1000 );
    camera.position.z = 0.5;
    camera.position.x = 0.2;
    camera.position.y = 0.15;

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0x000000 );

    //const particleNum = 1024 * 128;
    const particleNum = 1000;
    const particleSize = 3;

    // SPH Constants
    const REST_DENSITY = 1000.0;
    const L0 = 0.01;
    const N0 = 12;
    const PARTICLE_MASS = REST_DENSITY * L0**3;
    const SMOOTHING_LENGTH = (3.0 * N0 / (4 * MATH_PI))**(1.0/3.0) * L0;

    const WEIGHT_RHO_COEF = 315.0 / (64.0 * MATH_PI * SMOOTHING_LENGTH**3);
    const WEIGHT_PRESSURE_COEF = 45.0 / (MATH_PI * SMOOTHING_LENGTH**4);
    const WEIGHT_VISCOSITY_COEF = 45.0 / (MATH_PI * SMOOTHING_LENGTH**5);

    const PRESSURE_STIFFNESS = 15.0; // 15.0
    const VISC = 3.0;

    // boundary condition ( penalty method )
    const PARTICLE_RADIUS = L0 / 2;
    const EPSIRON = 0.0001;
    const EXT_STIFF = 10000.0;
    const EXT_DAMP = 250.0;

    const DT = 0.0025;

    const pos_min = [0.0, 0.0, 0.0];
    const pos_max = [0.3, 0.3, 0.1];

    const particleArray = new Float32Array( particleNum * particleSize );
    const velocityArray = new Float32Array( particleNum * particleSize );
    const colorArray = new Float32Array( particleNum * 3 );
    const forceArray = new Float32Array( particleNum * particleSize );
    const densityArray = new Float32Array( particleNum );
    const pressureArray = new Float32Array( particleNum );

    var x = 0;
    var y = 0;
    var z = 0;

    for ( let i = 0; i < particleArray.length / particleSize; i ++ ) {
        particleArray[ i * particleSize + 0 ] = pos_min[0] + L0 * x;
        particleArray[ i * particleSize + 1 ] = pos_min[1] + L0 * y;
        particleArray[ i * particleSize + 2 ] = pos_min[2] + L0 * z;
        x++;
        if (x >= 6)
        {
            x = 0;
            z++;
        }
        if (z >= 10)
        {
            x = 0;
            z = 0;
            y++;
        }
    }

    for ( let i = 0; i < particleArray.length / particleSize; i ++ ) {
        velocityArray[ i * particleSize + 0 ] = 0.0;
        velocityArray[ i * particleSize + 1 ] = 0.0;
        velocityArray[ i * particleSize + 2 ] = 0.0;

        colorArray[ i * 3 + 0 ] = 0.0;
        colorArray[ i * 3 + 1 ] = 1.0;
        colorArray[ i * 3 + 2 ] = 0.0;
    }

    const particleAttribute = new THREE.InstancedBufferAttribute( particleArray, particleSize );
    const velocityAttribute = new THREE.BufferAttribute( velocityArray, particleSize );
    const colorAttribute = new THREE.InstancedBufferAttribute( colorArray, 3 );
    const forceAttribute = new THREE.BufferAttribute( forceArray, particleSize );
    const densityAttribute = new THREE.BufferAttribute( densityArray, 1 );
    const pressureAttribute = new THREE.BufferAttribute( pressureArray, 1 );

    const particleBuffer = new WebGPUStorageBuffer( 'particle', particleAttribute );
    const velocityBuffer = new WebGPUStorageBuffer( 'velocity', velocityAttribute );
    const forceBuffer = new WebGPUStorageBuffer( 'force', forceAttribute );
    const densityBuffer = new WebGPUStorageBuffer( 'density', densityAttribute );
    const pressureBuffer = new WebGPUStorageBuffer( 'pressure', pressureAttribute );
        
    const computeBindings_density_pressure = [
        particleBuffer,
        densityBuffer,
        pressureBuffer
    ];

    const computeBindings_force = [
        particleBuffer,
        velocityBuffer,
        forceBuffer,
        densityBuffer,
        pressureBuffer
    ];

    const computeBindings_integrate = [
        particleBuffer,
        velocityBuffer,
        forceBuffer
    ];

    const computeShader_density_pressure = `
    #version 450

    // constants
    #define PARTICLE_NUM ${particleNum}
    #define PARTICLE_SIZE ${particleSize}
    const float mass = ${PARTICLE_MASS};
    const float h = ${SMOOTHING_LENGTH};

    const float rest_density = ${REST_DENSITY};
    const float pressure_stiffness = ${PRESSURE_STIFFNESS};
    const float w_rho_coef = ${WEIGHT_RHO_COEF};

    layout(set = 0, binding = 0) buffer Particle {
        float particle[ PARTICLE_NUM * PARTICLE_SIZE ];
    } particle;

    layout(set = 0, binding = 1) buffer Density {
        float density[ PARTICLE_NUM ];
    };

    layout(set = 0, binding = 2) buffer Pressure {
        float pressure[ PARTICLE_NUM ];
    };
    
    void main()
    {
        uint i = gl_GlobalInvocationID.x;
        if ( i >= PARTICLE_NUM ) { return; }

        // compute density
        float density_sum = 0.0;
        for (uint j = 0; j < PARTICLE_NUM; j++)
        {
            vec3 delta = vec3(
                particle.particle[ i * PARTICLE_SIZE + 0 ] - particle.particle[ j * PARTICLE_SIZE + 0 ],
                particle.particle[ i * PARTICLE_SIZE + 1 ] - particle.particle[ j * PARTICLE_SIZE + 1 ],
                particle.particle[ i * PARTICLE_SIZE + 2 ] - particle.particle[ j * PARTICLE_SIZE + 2 ]
            );

            float r = length(delta);
        
            if (r < h)
            {
                float rh = r / h;
                float rh2 = 1 - rh * rh;
                density_sum += mass * w_rho_coef * rh2 * rh2 * rh2;
            }
        }
        density[i] = density_sum;

        // compute pressure
        pressure[i] = max(pressure_stiffness * (density[i] - rest_density), 0.0);
    }
    `;

    const computeShader_force = `
    #version 450

    #define PARTICLE_NUM ${particleNum}
    #define PARTICLE_SIZE ${particleSize}
    // constants
    const float mass = ${PARTICLE_MASS};
    const float h = ${SMOOTHING_LENGTH};

    const float visc = ${VISC};
    const float w_pressure_coef = ${WEIGHT_PRESSURE_COEF};
    const float w_visc_coef = ${WEIGHT_VISCOSITY_COEF};

    const vec3 gravity = vec3(0, -9.8, 0.0);

    layout(set = 0, binding = 0) buffer Particle {
        float particle[ PARTICLE_NUM * PARTICLE_SIZE ];
    } particle;

    layout(set = 0, binding = 1) buffer Velocity {
        float velocity[ PARTICLE_NUM * PARTICLE_SIZE ];
    } velocity;

    layout(set = 0, binding = 2) buffer Force {
        float force[ PARTICLE_NUM * PARTICLE_SIZE ];
    } force;

    layout(set = 0, binding = 3) buffer Density {
        float density[ PARTICLE_NUM ];
    };

    layout(set = 0, binding = 4) buffer Pressure {
        float pressure[ PARTICLE_NUM ];
    };

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if ( i >= PARTICLE_NUM ) { return; }

        // forces
        vec3 pressure_force = vec3(0, 0, 0);
        vec3 viscosity_force = vec3(0, 0, 0);

        for (uint j = 0; j < PARTICLE_NUM; j++)
        {
            if (i == j)
            {
                continue;
            }
            vec3 delta = vec3(
                particle.particle[ i * PARTICLE_SIZE + 0 ] - particle.particle[ j * PARTICLE_SIZE + 0 ],
                particle.particle[ i * PARTICLE_SIZE + 1 ] - particle.particle[ j * PARTICLE_SIZE + 1 ],
                particle.particle[ i * PARTICLE_SIZE + 2 ] - particle.particle[ j * PARTICLE_SIZE + 2 ]
            );

            float r = length(delta);

            if (r < h)
            {
                float rh = 1 - r / h;

                pressure_force += 0.5 * mass * (pressure[i] + pressure[j]) / density[j] *
                    w_pressure_coef * rh * rh * normalize(delta);

                vec3 vji = vec3(
                    velocity.velocity[ j * PARTICLE_SIZE + 0 ] - velocity.velocity[ i * PARTICLE_SIZE + 0 ],
                    velocity.velocity[ j * PARTICLE_SIZE + 1 ] - velocity.velocity[ i * PARTICLE_SIZE + 1 ],
                    velocity.velocity[ j * PARTICLE_SIZE + 2 ] - velocity.velocity[ i * PARTICLE_SIZE + 2 ]
                );
                viscosity_force += visc * mass * vji / density[j] * w_visc_coef * rh;
            }
        }

        vec3 forces = ( pressure_force + viscosity_force ) / density[ i ] + gravity;
        //vec3 forces = gravity;

        force.force[ i * PARTICLE_SIZE + 0 ] = forces.x;
        force.force[ i * PARTICLE_SIZE + 1 ] = forces.y;
        force.force[ i * PARTICLE_SIZE + 2 ] = forces.z;
    }
    `;

    const computeShader_integrate = `
    #version 450
    #define PARTICLE_NUM ${particleNum}
    #define PARTICLE_SIZE ${particleSize}
    #define ROOM_SIZE 4.0
    // constants
    const float TIME_STEP = ${DT};
    const float radius = ${PARTICLE_RADIUS};
    const float epsiron = ${EPSIRON};
    const float extstiff = ${EXT_STIFF};
    const float extdamp = ${EXT_DAMP};

    const vec3 min_position = vec3(${pos_min[0]}, ${pos_min[1]}, ${pos_min[2]});
    const vec3 max_position = vec3(${pos_max[0]}, ${pos_max[1]}, ${pos_max[2]});

    layout(set = 0, binding = 0) buffer Particle {
        float particle[ PARTICLE_NUM * PARTICLE_SIZE ];
    } particle;

    layout(set = 0, binding = 1) buffer Velocity {
        float velocity[ PARTICLE_NUM * PARTICLE_SIZE ];
    } velocity;

    layout(set = 0, binding = 2) buffer Force {
        float force[ PARTICLE_NUM * PARTICLE_SIZE ];
    } force;

    void main() {
        uint index = gl_GlobalInvocationID.x;
        if ( index >= PARTICLE_NUM ) { return; }

        vec3 forces = vec3(
            force.force[ index * PARTICLE_SIZE + 0 ],
            force.force[ index * PARTICLE_SIZE + 1 ],
            force.force[ index * PARTICLE_SIZE + 2 ]
        );
        vec3 current_pos = vec3(
            particle.particle[ index * PARTICLE_SIZE + 0 ],
            particle.particle[ index * PARTICLE_SIZE + 1 ],
            particle.particle[ index * PARTICLE_SIZE + 2 ]
        );
        vec3 current_vel = vec3(
            velocity.velocity[ index * PARTICLE_SIZE + 0 ],
            velocity.velocity[ index * PARTICLE_SIZE + 1 ],
            velocity.velocity[ index * PARTICLE_SIZE + 2 ]
        );

        vec3 diff_min = 2.0 * radius - (current_pos - min_position);
        vec3 diff_max = 2.0 * radius - (max_position - current_pos);

        // boundary conditions
        // X
        if ( diff_min.x > epsiron ) {
            vec3 normal = vec3( 1.0, 0.0, 0.0 );
            float adj = extstiff * diff_min.x - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        if ( diff_max.x > epsiron ) {
            vec3 normal = vec3( -1.0, 0.0, 0.0 );
            float adj = extstiff * diff_max.x - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        // Y
        if ( diff_min.y > epsiron ) {
            vec3 normal = vec3( 0.0, 1.0, 0.0 );
            float adj = extstiff * diff_min.y - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        if ( diff_max.y > epsiron ) {
            vec3 normal = vec3( 0.0, -1.0, 0.0 );
            float adj = extstiff * diff_max.y - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        // Z
        if ( diff_min.z > epsiron ) {
            vec3 normal = vec3( 0.0, 0.0, 1.0 );
            float adj = extstiff * diff_min.z - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        if ( diff_max.z > epsiron ) {
            vec3 normal = vec3( 0.0, 0.0, -1.0 );
            float adj = extstiff * diff_max.z - extdamp * dot( normal, current_vel );
            forces += adj * normal;
        }
        
        vec3 new_velocity = current_vel + forces * TIME_STEP;
        vec3 new_position = current_pos + new_velocity * TIME_STEP;

        velocity.velocity[ index * PARTICLE_SIZE + 0 ] = new_velocity.x;
        velocity.velocity[ index * PARTICLE_SIZE + 1 ] = new_velocity.y;
        velocity.velocity[ index * PARTICLE_SIZE + 2 ] = new_velocity.z;
                        
        particle.particle[ index * PARTICLE_SIZE + 0 ] = new_position.x;
        particle.particle[ index * PARTICLE_SIZE + 1 ] = new_position.y;
        particle.particle[ index * PARTICLE_SIZE + 2 ] = new_position.z;
    }
    `;

    computeParams_density_pressure.push( {
        num: particleNum,
        shader: computeShader_density_pressure,
        bindings: computeBindings_density_pressure
    } );

    computeParams_force.push( {
        num: particleNum,
        shader: computeShader_force,
        bindings: computeBindings_force
    } );

    computeParams_integrate.push( {
        num: particleNum,
        shader: computeShader_integrate,
        bindings: computeBindings_integrate
    } );

    const boxSize = 0.002;
    const boxGeometry = new THREE.BoxBufferGeometry( boxSize, boxSize, boxSize );
    const geometry = new THREE.InstancedBufferGeometry()
        .setAttribute( 'position', boxGeometry.getAttribute( 'position' ) )
        .setAttribute( 'instancePosition', particleAttribute )
        .setAttribute( 'instanceColor', colorAttribute );
    geometry.setIndex( boxGeometry.getIndex() );
    geometry.instanceCount = particleNum;

    const material = new THREE.RawShaderMaterial( {
        vertexShader: `
        #version 450

        #define PARTICLE_NUM ${particleNum}
        #define PARTICLE_SIZE ${particleSize}

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 instancePosition;
        layout(location = 2) in vec3 instanceColor;

        layout(location = 0) out vec3 vColor;

        layout(set = 0, binding = 0) uniform ModelUniforms {
            mat4 modelMatrix;
            mat4 modelViewMatrix;
            mat3 normalMatrix;
        } modelUniforms;

        layout(set = 0, binding = 1) uniform CameraUniforms {
            mat4 projectionMatrix;
            mat4 viewMatrix;
        } cameraUniforms;

        void main(){
            vColor = instanceColor;
            gl_Position = cameraUniforms.projectionMatrix * modelUniforms.modelViewMatrix * vec4( position + instancePosition, 1.0 );
        }
        `,
        
        fragmentShader: `
        #version 450
        
        #define PARTICLE_NUM ${particleNum}

        layout(location = 0) in vec3 vColor;
        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = vec4( vColor, 1.0 );
        }
        `
    } );

    const bindings = [];

    const modelViewUniform = new Matrix4Uniform( 'modelMatrix' );
    const modelViewMatrixUniform = new Matrix4Uniform( 'modelViewMatrix' );
    const normalMatrixUniform = new Matrix3Uniform( 'normalMatrix' );

    const modelGroup = new WebGPUUniformsGroup( 'modelUniforms' );
    modelGroup.addUniform( modelViewUniform );
    modelGroup.addUniform( modelViewMatrixUniform );
    modelGroup.addUniform( normalMatrixUniform );
    modelGroup.setOnBeforeUpdate( function ( object/*, camera */ ) {
        modelViewUniform.setValue( object.matrixWorld );
        modelViewMatrixUniform.setValue( object.modelViewMatrix );
        normalMatrixUniform.setValue( object.normalMatrix );
    } );

    const projectionMatrixUniform = new Matrix4Uniform( 'projectionMatrix' );
    const viewMatrixUniform = new Matrix4Uniform( 'viewMatrix' );

    const cameraGroup = new WebGPUUniformsGroup( 'cameraUniforms' );
    cameraGroup.addUniform( projectionMatrixUniform );
    cameraGroup.addUniform( viewMatrixUniform );
    cameraGroup.setOnBeforeUpdate( function ( object, camera ) {
        projectionMatrixUniform.setValue( camera.projectionMatrix );
        viewMatrixUniform.setValue( camera.matrixWorldInverse );

    } );

    bindings.push( modelGroup );
    bindings.push( cameraGroup );

    material.bindings = bindings;

    const mesh = new THREE.Mesh( geometry, material );
    scene.add( mesh );
                
    // Box wireframe
    // front
    var geometryLine = new THREE.BufferGeometry().setFromPoints( [
        new THREE.Vector3( 0.0, 0.0, 0.1 ),
        new THREE.Vector3( 0.3, 0.0, 0.1 ),
        new THREE.Vector3( 0.3, 0.3, 0.1 ),
        new THREE.Vector3( 0.0, 0.3, 0.1 ),
        new THREE.Vector3( 0.0, 0.0, 0.1 ),
    ] );
    var materialLine = new THREE.LineBasicMaterial();
    var line = new THREE.Line( geometryLine, materialLine );
    scene.add( line );
    // left
    var geometryLine = new THREE.BufferGeometry().setFromPoints( [
        new THREE.Vector3( 0.0, 0.0, 0.1 ),
        new THREE.Vector3( 0.0, 0.0, 0.0 ),
        new THREE.Vector3( 0.0, 0.3, 0.0 ),
        new THREE.Vector3( 0.0, 0.3, 0.1 ),
        new THREE.Vector3( 0.0, 0.0, 0.1 ),
    ] );
    var materialLine = new THREE.LineBasicMaterial();
    var line = new THREE.Line( geometryLine, materialLine );
    scene.add( line );
    // back
    var geometryLine = new THREE.BufferGeometry().setFromPoints( [
        new THREE.Vector3( 0.0, 0.0, 0.0 ),
        new THREE.Vector3( 0.3, 0.0, 0.0 ),
        new THREE.Vector3( 0.3, 0.3, 0.0 ),
        new THREE.Vector3( 0.0, 0.3, 0.0 ),
        new THREE.Vector3( 0.0, 0.0, 0.0 ),
    ] );
    var materialLine = new THREE.LineBasicMaterial();
    var line = new THREE.Line( geometryLine, materialLine );
    scene.add( line );
    // right
    var geometryLine = new THREE.BufferGeometry().setFromPoints( [
        new THREE.Vector3(  0.3, 0.0, 0.1 ),
        new THREE.Vector3(  0.3, 0.0, 0.0 ),
        new THREE.Vector3(  0.3, 0.3, 0.0 ),
        new THREE.Vector3(  0.3, 0.3, 0.1 ),
        new THREE.Vector3(  0.3, 0.0, 0.1 ),
    ] );
    var materialLine = new THREE.LineBasicMaterial();
    var line = new THREE.Line( geometryLine, materialLine );
    scene.add( line );

    renderer = new WebGPURenderer();
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    window.addEventListener( 'resize', onWindowResize, false );

    return renderer.init();

} // init()

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {
    requestAnimationFrame( animate );

    renderer.compute( computeParams_density_pressure );
    renderer.compute( computeParams_force );
    renderer.compute( computeParams_integrate );
    renderer.render( scene, camera );

}

function error( error ) {
    console.error( error );

}

</script>
</body>
</html>
